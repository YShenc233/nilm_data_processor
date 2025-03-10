#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import os
import logging
import time
from datetime import datetime
from tqdm import tqdm
import sys
import argparse
import re
import hashlib

# 设置命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser(description="NILM Dataset Preprocessing Tool")
    
    # 基本参数
    parser.add_argument('--dataset', type=str, required=True, choices=['redd', 'ukdale', 'refit'],
                        help='Dataset to process: redd, ukdale, or refit')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing raw dataset')
    parser.add_argument('--method', type=str, default='method1', 
                        help='Processing method identifier (e.g., method1, method2)')
    
    # 处理模式
    parser.add_argument('--mode', type=str, default='combined', choices=['combined', 'separate', 'both'],
                        help='Output mode: combined (one file for all appliances), separate (one file per appliance), or both')
    
    # 选择房屋和电器
    parser.add_argument('--houses', type=str, default='',
                        help='House indices to process, comma separated (e.g., "1,2,3"). If empty, use default for dataset')
    parser.add_argument('--appliances', type=str, default='',
                        help='Appliances to process, comma separated. If empty, use default for dataset')
    parser.add_argument('--process_all_appliances', action='store_true',
                        help='Process all available appliances in dataset')
    
    # 数据处理参数
    parser.add_argument('--sampling', type=str, default='',
                        help='Resampling interval (e.g., "6s"). If empty, use default for dataset')
    parser.add_argument('--normalize', type=str, default='mean', choices=['mean', 'minmax', 'none'],
                        help='Normalization method')
    parser.add_argument('--validation_size', type=float, default=0.1,
                        help='Fraction of data to use for validation')
    parser.add_argument('--window_size', type=int, default=480,
                        help='Window size for data processing')
    parser.add_argument('--window_stride', type=int, default=120,
                        help='Window stride for data processing')
    
    # 日志级别
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    # 压缩选项
    parser.add_argument('--compress', action='store_true',
                        help='Compress output files to save space')
    
    # 短文件名选项
    parser.add_argument('--short_names', action='store_true',
                        help='Use short file names for combined mode output')
    
    # 跳过已处理文件
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip processing if output files already exist')
    
    return parser.parse_args()

# 设置日志
def setup_logger(name, log_file, level=logging.INFO):
    """设置应用和文件日志"""
    # 创建日志目录
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 防止重复日志
    if logger.handlers:
        return logger
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 创建短名称
def create_short_name(appliances):
    """为多个电器创建短名称，使用哈希值确保唯一性"""
    appliance_str = "_".join(sorted(appliances))
    hash_obj = hashlib.md5(appliance_str.encode())
    hash_str = hash_obj.hexdigest()[:8]
    return f"combined_{hash_str}"

class NILMDataProcessor:
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.method = args.method
        
        # 获取当前工作目录的绝对路径
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 设置数据目录
        self.data_dir = os.path.abspath(args.data_dir)
        
        # 设置项目基础目录结构
        self.project_dir = os.path.join(self.current_dir, "data")
        self.raw_dir = os.path.join(self.project_dir, "raw")
        self.processed_dir = os.path.join(self.project_dir, "processed")
        self.logs_dir = os.path.join(self.project_dir, "logs")
        self.readme_dir = os.path.join(self.project_dir, "readme")
        
        # 确保基础目录存在
        for directory in [self.project_dir, self.raw_dir, self.processed_dir, self.logs_dir, self.readme_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # 设置特定方法的输出目录
        self.method_dir = os.path.join(self.processed_dir, self.method)
        self.output_dir = os.path.join(self.method_dir, self.dataset)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置特定数据集的日志目录
        self.dataset_logs_dir = os.path.join(self.logs_dir, self.dataset)
        os.makedirs(self.dataset_logs_dir, exist_ok=True)
        
        # 设置日志文件名，包含方法和时间戳
        log_filename = f"{self.dataset}_{self.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file = os.path.join(self.dataset_logs_dir, log_filename)
        
        # 设置日志
        level = logging.DEBUG if args.verbose else logging.INFO
        self.logger = setup_logger(f'{self.dataset}_preprocessing', log_file, level)
        
        # 其他参数
        self.compress = args.compress
        self.short_names = args.short_names
        self.skip_existing = args.skip_existing
        
        # 设置房屋和电器
        self._setup_houses_and_appliances()
        
        # 设置采样率
        self._setup_sampling_rate()
        
        # 设置数据处理参数
        self.normalize = args.normalize
        self.validation_size = args.validation_size
        self.window_size = args.window_size
        self.window_stride = args.window_stride
        
        # 初始化电器特定参数
        self._initialize_appliance_params()
        
        # 显示处理信息
        self._log_processing_info()
    
    def _setup_houses_and_appliances(self):
        """设置要处理的房屋和电器"""
        # 设置默认值
        if self.dataset == 'redd':
            default_houses = [1, 2, 3, 4, 5, 6]
            default_appliances = ['refrigerator', 'dishwasher', 'microwave', 'washer_dryer']
            self.house_field = 'house_'
            self.aggregate_name = 'aggregate'
        elif self.dataset == 'ukdale':
            default_houses = [1, 2, 3, 4, 5]
            default_appliances = ['kettle', 'fridge', 'washing_machine', 'microwave', 'dishwasher', 'toaster']
            self.house_field = 'house_'
            self.aggregate_name = 'aggregate'
        else:  # refit
            default_houses = [2, 3, 5, 16]
            default_appliances = ['Washing_Machine', 'Fridge-Freezer', 'TV', 'Kettle', 'Microwave', 'Dishwasher']
            self.house_field = 'House'  # 注意Refit使用'House'而不是'house_'
            self.aggregate_name = 'Aggregate'
        
        # 解析房屋参数
        if self.args.houses:
            self.houses = [int(h) for h in self.args.houses.split(',')]
        else:
            self.houses = default_houses
            self.logger.info(f"Using default houses for {self.dataset}: {self.houses}")
        
        # 解析电器参数
        if self.args.appliances:
            self.appliances = [a.strip() for a in self.args.appliances.split(',')]
        else:
            self.appliances = default_appliances
            self.logger.info(f"Using default appliances for {self.dataset}: {self.appliances}")
    
    def _setup_sampling_rate(self):
        """设置采样率"""
        if self.args.sampling:
            self.sampling = self.args.sampling
        else:
            # 默认采样率
            if self.dataset == 'redd':
                self.sampling = '6s'
            elif self.dataset == 'ukdale':
                self.sampling = '6s'
            else:  # refit
                self.sampling = '7s'
            self.logger.info(f"Using default sampling rate for {self.dataset}: {self.sampling}")
    
    def _initialize_appliance_params(self):
        """初始化电器特定参数（阈值、最小开关时间等）"""
        if self.dataset == 'redd':
            self.cutoff = {
                'aggregate': 6000,
                'refrigerator': 400,
                'washer_dryer': 3500,
                'microwave': 1800,
                'dishwasher': 1200
            }
            self.threshold = {
                'refrigerator': 50,
                'washer_dryer': 20,
                'microwave': 200,
                'dishwasher': 10
            }
            self.min_on = {
                'refrigerator': 10,
                'washer_dryer': 300,
                'microwave': 2,
                'dishwasher': 300
            }
            self.min_off = {
                'refrigerator': 2,
                'washer_dryer': 26,
                'microwave': 5,
                'dishwasher': 300
            }
        elif self.dataset == 'ukdale':
            self.cutoff = {
                'aggregate': 6000,
                'kettle': 3100,
                'fridge': 300,
                'washing_machine': 2500,
                'microwave': 3000,
                'dishwasher': 2500,
                'toaster': 3100
            }
            self.threshold = {
                'kettle': 2000,
                'fridge': 50,
                'washing_machine': 20,
                'microwave': 200,
                'dishwasher': 10,
                'toaster': 1000
            }
            self.min_on = {
                'kettle': 2,
                'fridge': 10,
                'washing_machine': 300,
                'microwave': 2,
                'dishwasher': 300,
                'toaster': 2000
            }
            self.min_off = {
                'kettle': 0,
                'fridge': 2,
                'washing_machine': 26,
                'microwave': 5,
                'dishwasher': 300,
                'toaster': 0
            }
        else:  # refit
            self.cutoff = {
                'Aggregate': 10000,
                'Kettle': 3000,
                'Fridge-Freezer': 1700,
                'Washing_Machine': 2500,
                'Microwave': 1300,
                'Dishwasher': 2500,
                'TV': 80
            }
            self.threshold = {
                'Kettle': 2000,
                'Fridge-Freezer': 5,
                'Washing_Machine': 20,
                'Microwave': 200,
                'Dishwasher': 10,
                'TV': 10
            }
            self.min_on = {
                'Kettle': 2,
                'Fridge-Freezer': 10,
                'Washing_Machine': 10,
                'Microwave': 2,
                'Dishwasher': 300,
                'TV': 2
            }
            self.min_off = {
                'Kettle': 0,
                'Fridge-Freezer': 2,
                'Washing_Machine': 26,
                'Microwave': 5,
                'Dishwasher': 300,
                'TV': 0
            }
        
        # 添加默认值，以防某些电器没有定义参数
        for appliance in self.appliances:
            if appliance not in self.threshold:
                self.threshold[appliance] = 20
                self.logger.warning(f"No threshold defined for {appliance}, using default value: 20")
            if appliance not in self.min_on:
                self.min_on[appliance] = 10
                self.logger.warning(f"No min_on defined for {appliance}, using default value: 10")
            if appliance not in self.min_off:
                self.min_off[appliance] = 5
                self.logger.warning(f"No min_off defined for {appliance}, using default value: 5")
            if appliance not in self.cutoff:
                self.cutoff[appliance] = 2000
                self.logger.warning(f"No cutoff defined for {appliance}, using default value: 2000")
    
    def _log_processing_info(self):
        """记录处理信息"""
        self.logger.info("=" * 50)
        self.logger.info(f"NILM Dataset Preprocessing - {self.dataset.upper()} - Method: {self.method}")
        self.logger.info(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Parameters:")
        self.logger.info(f"  - Houses: {self.houses}")
        self.logger.info(f"  - Appliances: {self.appliances}")
        self.logger.info(f"  - Sampling: {self.sampling}")
        self.logger.info(f"  - Normalization: {self.normalize}")
        self.logger.info(f"  - Output mode: {self.args.mode}")
        self.logger.info(f"  - Validation size: {self.validation_size}")
        self.logger.info(f"  - Window size: {self.window_size}")
        self.logger.info(f"  - Window stride: {self.window_stride}")
        self.logger.info(f"Paths:")
        self.logger.info(f"  - Working Directory: {self.current_dir}")
        self.logger.info(f"  - Input Data: {self.data_dir}")
        self.logger.info(f"  - Output Directory: {self.output_dir}")
        self.logger.info(f"  - Logs Directory: {self.dataset_logs_dir}")
        self.logger.info("=" * 50)
    
    def process_dataset(self):
        """处理选定的数据集"""
        start_time = time.time()
        
        self.logger.info(f"Stage 1/4: Loading and processing {self.dataset.upper()} dataset...")
        
        if self.dataset == 'redd':
            data, processed_appliances = self._process_redd()
        elif self.dataset == 'ukdale':
            data, processed_appliances = self._process_ukdale()
        else:  # refit
            data, processed_appliances = self._process_refit()
        
        if data is None or data.empty:
            self.logger.error("No data found or processed. Check your data path and parameters.")
            return
        
        self.logger.info(f"Successfully loaded data with {len(data)} samples and {len(processed_appliances)} appliances")
        
        # 计算每个电器的状态
        self.logger.info("Stage 2/4: Computing appliance states...")
        with tqdm(total=len(processed_appliances), desc="Computing status") as pbar:
            for appliance in processed_appliances:
                if appliance in self.threshold:
                    status = self._compute_status(data, appliance)
                    data[f"{appliance}_status"] = status
                    pbar.update(1)
                else:
                    self.logger.warning(f"No threshold parameters for {appliance}, skipping status computation")
        
        # 标准化数据
        if self.normalize != 'none':
            self.logger.info(f"Stage 3/4: Normalizing data...")
            data, norm_params = self._normalize_data(data)
        else:
            self.logger.info(f"Stage 3/4: Skipping normalization (method='none')")
            norm_params = None
        
        # 保存处理后的数据
        self.logger.info("Stage 4/4: Saving processed data...")
        if self.args.mode == 'combined' or self.args.mode == 'both':
            self._save_combined_data(data, processed_appliances, norm_params)
        
        if self.args.mode == 'separate' or self.args.mode == 'both':
            self._save_separate_data(data, processed_appliances, norm_params)
        
        elapsed_time = time.time() - start_time
        self.logger.info("=" * 50)
        self.logger.info(f"Preprocessing complete! Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        self.logger.info(f"Processed {len(data)} samples across {len(processed_appliances)} appliances")
        self.logger.info("=" * 50)
    
    def _process_redd(self):
        """处理REDD数据集"""
        entire_data = None
        processed_houses = 0
        available_appliances = set()
        
        # 创建房屋处理进度条
        house_pbar = tqdm(self.houses, desc="Processing houses", unit="house")
        
        for house_id in house_pbar:
            house_pbar.set_description(f"Processing house {house_id}")
            if house_id not in [1, 2, 3, 4, 5, 6]:
                self.logger.warning(f"Unsupported house_id: {house_id}, skipping...")
                continue

            house_folder = os.path.join(self.data_dir, f'house_{house_id}')
            
            try:
                # 检查必要的文件是否存在
                if not os.path.exists(house_folder):
                    self.logger.warning(f"House folder {house_folder} not found, skipping...")
                    continue
                    
                label_file = os.path.join(house_folder, 'labels.dat')
                main1_file = os.path.join(house_folder, 'channel_1.dat')
                main2_file = os.path.join(house_folder, 'channel_2.dat')
                
                if not (os.path.exists(label_file) and os.path.exists(main1_file) and os.path.exists(main2_file)):
                    self.logger.warning(f"Required files for house {house_id} not found, skipping...")
                    continue
                
                # 读取标签和主电表数据
                self.logger.info(f"Reading data files for house {house_id}")
                house_label = pd.read_csv(label_file, sep=' ', header=None)
                main_1 = pd.read_csv(main1_file, sep=' ', header=None)
                main_2 = pd.read_csv(main2_file, sep=' ', header=None)
                
                # 使用现代Pandas方法处理时间戳
                # 创建基本数据框架
                house_data = pd.DataFrame()
                house_data['time'] = pd.to_datetime(main_1[0], unit='s')
                house_data['aggregate'] = main_1[1] + main_2[1]
                
                # 获取电器列表
                appliance_list = house_label.iloc[:, 1].values
                app_index_dict = defaultdict(list)
                
                # 找到所有可用电器
                valid_appliances = ['dishwasher', 'refrigerator', 'microwave', 'washer_dryer']
                for app in appliance_list:
                    if app in valid_appliances:
                        available_appliances.add(app)
                
                # 如果process_all_appliances为True，更新self.appliances
                if self.args.process_all_appliances:
                    self.appliances = list(available_appliances)
                    self.logger.info(f"Processing all available appliances: {self.appliances}")
                
                # 找到每个目标电器的通道
                found_appliances = []
                for appliance in self.appliances:
                    try:
                        idx = appliance_list.tolist().index(appliance)
                        app_index_dict[appliance].append(idx + 1)
                        found_appliances.append(appliance)
                        self.logger.info(f"Found {appliance} in house {house_id} at channel {idx+1}")
                    except ValueError:
                        app_index_dict[appliance].append(-1)
                        self.logger.warning(f"{appliance} not found in house {house_id}")
                
                # 如果没有找到任何目标电器，跳过这个房屋
                if not found_appliances:
                    self.logger.warning(f"No target appliances found in house {house_id}, skipping...")
                    continue
                
                self.logger.info(f"Processing appliance data for house {house_id}")
                
                # 处理每个电器，逐一添加到主数据框架
                appliance_pbar = tqdm(found_appliances, desc=f"  House {house_id} appliances", leave=False)
                for appliance in appliance_pbar:
                    appliance_pbar.set_description(f"  Processing {appliance} in house {house_id}")
                    
                    if app_index_dict[appliance][0] == -1:
                        # 如果没有找到电器，用0填充
                        house_data[appliance] = 0
                        self.logger.info(f"No {appliance} found in house {house_id}, filled with zeros")
                    else:
                        # 读取电器数据
                        channel_idx = app_index_dict[appliance][0]
                        channel_file = os.path.join(house_folder, f'channel_{channel_idx}.dat')
                        
                        try:
                            app_data = pd.read_csv(channel_file, sep=' ', header=None)
                            app_data.columns = ['time', appliance]
                            app_data['time'] = pd.to_datetime(app_data['time'], unit='s')
                            
                            # 合并电器数据，使用基于时间的合并
                            house_data = pd.merge(house_data, app_data, on='time', how='outer')
                            self.logger.info(f"Loaded {appliance} data from channel {channel_idx}")
                        except Exception as e:
                            self.logger.error(f"Error loading data for {appliance} at channel {channel_idx}: {str(e)}")
                            house_data[appliance] = 0
                    
                    # 处理多通道电器（如果有）
                    if len(app_index_dict[appliance]) > 1:
                        for idx in app_index_dict[appliance][1:]:
                            try:
                                channel_file = os.path.join(house_folder, f'channel_{idx}.dat')
                                add_data = pd.read_csv(channel_file, sep=' ', header=None)
                                add_data.columns = ['time', f"{appliance}_add"]
                                add_data['time'] = pd.to_datetime(add_data['time'], unit='s')
                                
                                # 合并附加通道
                                house_data = pd.merge(house_data, add_data, on='time', how='outer')
                                
                                # 合并功率值
                                if f"{appliance}_add" in house_data.columns:
                                    house_data[appliance] = house_data[appliance].fillna(0) + house_data[f"{appliance}_add"].fillna(0)
                                    house_data = house_data.drop(columns=[f"{appliance}_add"])
                                    
                                self.logger.info(f"Merged additional channel {idx} for {appliance}")
                            except Exception as e:
                                self.logger.error(f"Error processing additional channel {idx} for {appliance}: {str(e)}")
                
                # 设置时间索引并重采样
                house_data = house_data.set_index('time')
                
                # 排序并处理缺失值
                house_data = house_data.sort_index()
                
                # 重采样并填充缺失值
                self.logger.info(f"Resampling house {house_id} data to {self.sampling} intervals")
                house_data = house_data.resample(self.sampling).mean().fillna(method='ffill', limit=30)
                
                self.logger.info(f"House {house_id} processing complete: {len(house_data)} samples")
                processed_houses += 1
                
                # 合并到整体数据
                if entire_data is None:
                    entire_data = house_data
                else:
                    entire_data = pd.concat([entire_data, house_data], axis=0)
                    
            except Exception as e:
                self.logger.error(f"Error processing house {house_id}: {str(e)}")
                continue
        
        # 清理和限制数据
        if entire_data is not None and not entire_data.empty:
            self.logger.info("Processing complete dataset")
            
            # 丢弃NA并移除无效数据
            initial_size = len(entire_data)
            entire_data = entire_data.dropna().copy()
            
            # 检查是否有数据剩余
            if entire_data.empty:
                self.logger.warning("All data was dropped when removing NA values")
                return None, []
            
            # 过滤和清理
            entire_data = entire_data[entire_data['aggregate'] > 0]  # 移除负值
            self.logger.info(f"Removed {initial_size - len(entire_data)} invalid rows")
            
            # 将小值设为0
            small_values = (entire_data < 5).sum().sum()
            entire_data[entire_data < 5] = 0
            self.logger.info(f"Set {small_values} small values (<5) to zero")
            
            # 限制数值范围
            cutoff_values = {'aggregate': self.cutoff.get('aggregate', 6000)}
            for app in entire_data.columns:
                if app != 'aggregate':
                    cutoff_values[app] = self.cutoff.get(app, 1000)
            
            # 单独应用截断以避免警告
            for col in entire_data.columns:
                entire_data[col] = entire_data[col].clip(lower=0, upper=cutoff_values.get(col, 1000))
            
            self.logger.info(f"Successfully processed {processed_houses} houses with {len(entire_data)} total samples")
            
            # 返回处理后的数据和实际处理的电器列表
            actual_appliances = [col for col in entire_data.columns if col != 'aggregate']
            return entire_data, actual_appliances
        else:
            self.logger.warning("No data was processed. Check house indices and appliance names.")
            return None, []
    
    def _process_ukdale(self):
        """处理UK-DALE数据集"""
        entire_data = None
        processed_houses = 0
        available_appliances = set()
        
        # 创建房屋处理进度条
        house_pbar = tqdm(self.houses, desc="Processing houses", unit="house")
        
        for house_id in house_pbar:
            house_pbar.set_description(f"Processing house {house_id}")
            if house_id not in [1, 2, 3, 4, 5]:
                self.logger.warning(f"Unsupported house_id: {house_id}, skipping...")
                continue
            
            try:
                house_folder = os.path.join(self.data_dir, f'house_{house_id}')
                
                # 检查必要的文件是否存在
                if not os.path.exists(house_folder):
                    self.logger.warning(f"House folder {house_folder} not found, skipping...")
                    continue
                    
                label_file = os.path.join(house_folder, 'labels.dat')
                main_file = os.path.join(house_folder, 'channel_1.dat')
                
                if not (os.path.exists(label_file) and os.path.exists(main_file)):
                    self.logger.warning(f"Required files for house {house_id} not found, skipping...")
                    continue
                
                # 读取标签和主电表数据
                self.logger.info(f"Reading data files for house {house_id}")
                house_label = pd.read_csv(label_file, sep=' ', header=None)    
                house_data = pd.read_csv(main_file, sep=' ', header=None)
                
                # 设置列名并转换时间
                house_data.columns = ['time', 'aggregate']
                house_data['time'] = pd.to_datetime(house_data['time'], unit='s')
                house_data = house_data.set_index('time').resample(self.sampling).mean().fillna(method='ffill', limit=30)
                
                # 获取电器列表
                appliance_list = house_label.iloc[:, 1].values
                app_index_dict = defaultdict(list)
                
                # 找到所有可用电器
                valid_appliances = ['dishwasher', 'fridge', 'microwave', 'washing_machine', 'kettle', 'toaster']
                for app in appliance_list:
                    if app in valid_appliances:
                        available_appliances.add(app)
                
                # 如果process_all_appliances为True，更新self.appliances
                if self.args.process_all_appliances:
                    self.appliances = list(available_appliances)
                    self.logger.info(f"Processing all available appliances: {self.appliances}")
                
                # 找到每个目标电器的通道
                found_appliances = []
                for appliance in self.appliances:
                    try:
                        idx = appliance_list.tolist().index(appliance)
                        app_index_dict[appliance].append(idx + 1)
                        found_appliances.append(appliance)
                        self.logger.info(f"Found {appliance} in house {house_id} at channel {idx+1}")
                    except ValueError:
                        app_index_dict[appliance].append(-1)
                        self.logger.warning(f"{appliance} not found in house {house_id}")
                
                # 如果没有找到任何目标电器，跳过这个房屋
                if not found_appliances:
                    self.logger.warning(f"No target appliances found in house {house_id}, skipping...")
                    continue
                
                self.logger.info(f"Processing appliance data for house {house_id}")
                
                # 处理每个电器的数据
                appliance_pbar = tqdm(found_appliances, desc=f"  House {house_id} appliances", leave=False)
                for appliance in appliance_pbar:
                    appliance_pbar.set_description(f"  Processing {appliance} in house {house_id}")
                    
                    channel_idx = app_index_dict[appliance][0]
                    if channel_idx == -1:
                        # 如果没有找到电器，用0填充
                        house_data[appliance] = np.zeros(len(house_data))
                        self.logger.info(f"No {appliance} found in house {house_id}, filled with zeros")
                    else:
                        # 读取电器数据
                        channel_path = os.path.join(house_folder, f'channel_{channel_idx}.dat')
                        appl_data = pd.read_csv(channel_path, sep=' ', header=None)
                        appl_data.columns = ['time', appliance]
                        appl_data['time'] = pd.to_datetime(appl_data['time'], unit='s')          
                        appl_data = appl_data.set_index('time').resample(self.sampling).mean().fillna(method='ffill', limit=30)   
                        
                        # 合并到主数据
                        before_merge = len(house_data)
                        house_data = pd.merge(house_data, appl_data, how='inner', left_index=True, right_index=True)
                        after_merge = len(house_data)
                        self.logger.info(f"Merged {appliance} data, rows before: {before_merge}, after: {after_merge}")
                
                self.logger.info(f"House {house_id} processing complete: {len(house_data)} samples")
                processed_houses += 1
                
                # 重置索引以便合并
                house_data = house_data.reset_index()
                
                # 合并到整体数据
                if entire_data is None:
                    entire_data = house_data
                else:
                    entire_data = pd.concat([entire_data, house_data], ignore_index=True)
                    
            except Exception as e:
                self.logger.error(f"Error processing house {house_id}: {str(e)}")
                continue
        
        # 清理和限制数据
        if entire_data is not None and not entire_data.empty:
            self.logger.info("Processing complete dataset")
            
            # 设置时间索引
            if 'time' in entire_data.columns:
                entire_data = entire_data.set_index('time')
            
            # 丢弃NA并移除无效数据
            initial_size = len(entire_data)
            entire_data = entire_data.dropna().copy()
            
            # 检查是否有数据剩余
            if entire_data.empty:
                self.logger.warning("All data was dropped when removing NA values")
                return None, []
            
            # 过滤和清理
            entire_data = entire_data[entire_data['aggregate'] > 0]  # 移除负值
            self.logger.info(f"Removed {initial_size - len(entire_data)} invalid rows")
            
            # 将小值设为0
            small_values = (entire_data < 5).sum().sum()
            entire_data[entire_data < 5] = 0
            self.logger.info(f"Set {small_values} small values (<5) to zero")
            
            # 限制数值范围
            cutoff_values = {'aggregate': self.cutoff.get('aggregate', 6000)}
            processed_appliances = [col for col in entire_data.columns if col != 'aggregate']
            for app in processed_appliances:
                cutoff_values[app] = self.cutoff.get(app, 3000)
            
            # 单独应用截断以避免警告
            for col in entire_data.columns:
                entire_data[col] = entire_data[col].clip(lower=0, upper=cutoff_values.get(col, 3000))
            
            self.logger.info(f"Successfully processed {processed_houses} houses with {len(entire_data)} total samples")
            
            return entire_data, processed_appliances
        else:
            self.logger.warning("No data was processed. Check house indices and appliance names.")
            return None, []
    
    def _process_refit(self):
        """处理REFIT数据集"""
        entire_data = None
        processed_houses = 0
        
        # 创建房屋处理进度条
        house_pbar = tqdm(self.houses, desc="Processing houses", unit="house")
        
        for house_idx in house_pbar:
            house_pbar.set_description(f"Processing house {house_idx}")
            
            try:
                filename = f'House{house_idx}.csv'
                labelname = f'House{house_idx}.txt'
                
                # 检查文件是否存在
                data_folder = os.path.join(self.data_dir, 'Data')
                labels_folder = os.path.join(self.data_dir, 'Labels')
                
                if not (os.path.exists(data_folder) and os.path.exists(labels_folder)):
                    self.logger.error(f"Expected 'Data' and 'Labels' subdirectories in {self.data_dir}")
                    return None, []
                
                house_data_path = os.path.join(data_folder, filename)
                house_label_path = os.path.join(labels_folder, labelname)
                
                if not (os.path.exists(house_data_path) and os.path.exists(house_label_path)):
                    self.logger.warning(f"Data or label file for House {house_idx} not found, skipping...")
                    continue
                
                # 读取标签
                with open(house_label_path) as f:
                    house_labels = f.readlines()
                
                # 解析标签
                house_labels = ['Time'] + house_labels[0].split(',')
                house_labels = [label.strip() for label in house_labels]
                
                # 找出所有可用电器
                available_appliances = [label for label in house_labels 
                                       if label not in ['Time', 'Aggregate', 'Issues']]
                
                # 如果process_all_appliances为True，更新self.appliances
                if self.args.process_all_appliances:
                    self.appliances = available_appliances
                    self.logger.info(f"Processing all available appliances: {self.appliances}")
                
                # 检查目标电器是否在此房屋中
                found_appliances = [app for app in self.appliances if app in house_labels]
                if not found_appliances:
                    self.logger.warning(f"None of the target appliances found in House {house_idx}, skipping...")
                    continue
                
                self.logger.info(f"Found appliances in House {house_idx}: {found_appliances}")
                
                # 读取房屋数据 - 使用tqdm显示进度
                self.logger.info(f"Reading data file for House {house_idx}")
                
                # 获取文件大小估计行数
                file_size = os.path.getsize(house_data_path)
                est_lines = file_size / 100  # 粗略估计
                
                # 读取数据并过滤
                house_data = pd.read_csv(house_data_path)
                self.logger.info(f"Loaded {len(house_data)} rows from House {house_idx}")
                
                # 转换时间戳
                self.logger.info("Converting timestamps...")
                house_data['Unix'] = pd.to_datetime(house_data['Unix'], unit='s')
                
                # 重命名列
                house_data = house_data.drop(labels=['Time'], axis=1)
                house_data.columns = house_labels
                house_data = house_data.set_index('Time')
                
                # 删除标记为有问题的数据
                issues_count = house_data[house_data['Issues'] == 1].shape[0] if 'Issues' in house_data.columns else 0
                self.logger.info(f"Removing {issues_count} rows with issues")
                if 'Issues' in house_data.columns:
                    idx_to_drop = house_data[house_data['Issues'] == 1].index
                    house_data = house_data.drop(index=idx_to_drop, axis=0)
                    house_data = house_data.drop(columns=['Issues'])
                
                # 只保留需要的列
                cols_to_keep = ['Aggregate'] + found_appliances
                house_data = house_data[cols_to_keep]
                
                # 重采样并填充缺失值
                self.logger.info(f"Resampling to {self.sampling} intervals")
                initial_len = len(house_data)
                house_data = house_data.resample(self.sampling).mean().fillna(method='ffill', limit=30)
                resampled_len = len(house_data)
                self.logger.info(f"Resampling: {initial_len} -> {resampled_len} rows")
                
                # 重置索引以便合并
                house_data = house_data.reset_index()
                
                # 合并数据
                if entire_data is None:
                    entire_data = house_data
                else:
                    entire_data = pd.concat([entire_data, house_data], ignore_index=True)
                
                self.logger.info(f"House {house_idx} processing complete: {len(house_data)} samples")
                processed_houses += 1
                
            except Exception as e:
                self.logger.error(f"Error processing house {house_idx}: {str(e)}")
                continue
        
        # 清理和限制数据
        if entire_data is not None and not entire_data.empty:
            self.logger.info("Processing complete dataset")
            
            # 设置时间索引
            if 'Time' in entire_data.columns:
                entire_data = entire_data.set_index('Time')
            
            # 丢弃NA并移除无效数据
            initial_size = len(entire_data)
            entire_data = entire_data.dropna().copy()
            
            # 检查是否有数据剩余
            if entire_data.empty:
                self.logger.warning("All data was dropped when removing NA values")
                return None, []
            
            # 过滤和清理
            entire_data = entire_data[entire_data['Aggregate'] > 0]  # 移除负值
            self.logger.info(f"Removed {initial_size - len(entire_data)} invalid rows")
            
            # 将小值设为0
            small_values = (entire_data < 5).sum().sum()
            entire_data[entire_data < 5] = 0
            self.logger.info(f"Set {small_values} small values (<5) to zero")
            
            # 限制数值范围
            cutoff_values = {'Aggregate': self.cutoff.get('Aggregate', 10000)}
            processed_appliances = [col for col in entire_data.columns if col != 'Aggregate']
            for app in processed_appliances:
                cutoff_values[app] = self.cutoff.get(app, 2500)
            
            # 单独应用截断以避免警告
            for col in entire_data.columns:
                entire_data[col] = entire_data[col].clip(lower=0, upper=cutoff_values.get(col, 2500))
            
            self.logger.info(f"Successfully processed {processed_houses} houses with {len(entire_data)} total samples")
            
            return entire_data, processed_appliances
        else:
            self.logger.warning("No data was processed. Check house indices and appliance names.")
            return None, []
    
    def _compute_status(self, data, appliance):
        """计算电器状态 (开/关)"""
        self.logger.info(f"Computing status for {appliance} using threshold {self.threshold[appliance]}")
        
        appliance_data = data[appliance].values
        
        # 基于阈值确定初始状态
        initial_status = appliance_data >= self.threshold[appliance]
        on_count = initial_status.sum()
        self.logger.info(f"Initial status: {on_count} ON points out of {len(initial_status)} ({on_count/len(initial_status)*100:.2f}%)")
        
        status_diff = np.diff(initial_status)
        events_idx = status_diff.nonzero()[0]
        events_idx += 1

        if initial_status[0]:
            events_idx = np.insert(events_idx, 0, 0)

        if initial_status[-1]:
            events_idx = np.insert(events_idx, events_idx.size, initial_status.size)

        # 重塑为开/关事件对
        if len(events_idx) % 2 != 0:
            self.logger.warning(f"Odd number of events: {len(events_idx)}, dropping last event")
            events_idx = events_idx[:-1]  # 确保有偶数个事件
        
        if len(events_idx) == 0:
            self.logger.warning(f"No ON/OFF events found for {appliance}")
            return np.zeros_like(appliance_data)
        
        events_idx = events_idx.reshape((-1, 2))
        initial_events = len(events_idx)
        self.logger.info(f"Found {initial_events} initial ON/OFF event pairs")
        
        on_events = events_idx[:, 0].copy()
        off_events = events_idx[:, 1].copy()
        
        # 应用最小开/关时间限制
        if len(on_events) > 0:
            # 过滤短暂的关闭状态
            off_duration = on_events[1:] - off_events[:-1]
            off_duration = np.insert(off_duration, 0, 1000)  # 确保第一个事件被保留
            valid_off = off_duration > self.min_off[appliance]
            
            filtered_on = on_events[valid_off]
            filtered_off = off_events[np.roll(valid_off, -1)]
            
            # 确保长度一致
            min_len = min(len(filtered_on), len(filtered_off))
            filtered_on = filtered_on[:min_len]
            filtered_off = filtered_off[:min_len]
            
            self.logger.info(f"After min_off filter: {len(filtered_on)} events (removed {len(on_events)-len(filtered_on)} events)")
            
            if len(filtered_on) > 0:
                # 过滤短暂的开启状态
                on_duration = filtered_off - filtered_on
                valid_on = on_duration >= self.min_on[appliance]
                
                on_events = filtered_on[valid_on]
                off_events = filtered_off[valid_on]
                self.logger.info(f"After min_on filter: {len(on_events)} events (removed {len(filtered_on)-len(on_events)} events)")

        # 创建状态数组
        status = np.zeros_like(appliance_data)
        for on, off in zip(on_events, off_events):
            if on < len(status) and off <= len(status):
                status[on:off] = 1
        
        final_on_count = status.sum()
        self.logger.info(f"Final status: {final_on_count} ON points ({final_on_count/len(status)*100:.2f}%)")
        if initial_events > 0:
            self.logger.info(f"Reduction from filtering: {initial_events} to {len(on_events)} events ({len(on_events)/initial_events*100:.2f}% kept)")
            
        return status
    
    def _normalize_data(self, data):
        """标准化数据"""
        self.logger.info(f"Normalizing data using '{self.normalize}' method")
        
        aggregate_column = self.aggregate_name  # 'aggregate' for REDD/UK-DALE, 'Aggregate' for REFIT
        
        if self.normalize == 'mean':
            # 均值标准化
            x_mean = data[aggregate_column].mean()
            x_std = data[aggregate_column].std()
            data[aggregate_column] = (data[aggregate_column] - x_mean) / x_std
            self.logger.info(f"Mean normalization - mean: {x_mean:.2f}, std: {x_std:.2f}")
            return data, {'mean': x_mean, 'std': x_std}
        elif self.normalize == 'minmax':
            # 最小-最大标准化
            x_min = data[aggregate_column].min()
            x_max = data[aggregate_column].max()
            data[aggregate_column] = (data[aggregate_column] - x_min) / (x_max - x_min)
            self.logger.info(f"Min-max normalization - min: {x_min:.2f}, max: {x_max:.2f}")
            return data, {'min': x_min, 'max': x_max}
        
        return data, None
    
    def _save_combined_data(self, data, processed_appliances, norm_params):
        """保存合并模式数据（所有电器在一个文件中）"""
        # 生成文件名
        if self.short_names:
            filename_base = f"{self.dataset}_{create_short_name(processed_appliances)}"
        else:
            filename_base = f"{self.dataset}_processed_{'_'.join(processed_appliances)}"
        
        # 检查是否存在已处理的文件
        output_file = os.path.join(self.output_dir, f"{filename_base}.csv")
        if self.skip_existing and os.path.exists(output_file):
            self.logger.info(f"File {output_file} already exists, skipping...")
            return
        
        # 保存数据
        compression = 'gzip' if self.compress else None
        extension = '.csv.gz' if self.compress else '.csv'
        
        output_file = os.path.join(self.output_dir, f"{filename_base}{extension}")
        data.to_csv(output_file, compression=compression)
        self.logger.info(f"Combined data saved to {output_file}")
        
        # 保存状态信息
        status_file = os.path.join(self.output_dir, f"{filename_base}_status{extension}")
        status_data = data[[col for col in data.columns if col.endswith('_status')]]
        status_data.to_csv(status_file, compression=compression)
        self.logger.info(f"Status data saved to {status_file}")
        
        # 保存标准化参数
        if norm_params:
            stats_file = os.path.join(self.output_dir, f"{filename_base}_stats.csv")
            pd.DataFrame([norm_params]).to_csv(stats_file, index=False)
            self.logger.info(f"Normalization stats saved to {stats_file}")
        
        # 保存数据信息
        metadata = {
            'dataset': self.dataset,
            'method': self.method,
            'houses': self.houses,
            'appliances': processed_appliances,
            'total_samples': len(data),
            'sampling_rate': self.sampling,
            'normalization': self.normalize,
            'window_size': self.window_size,
            'window_stride': self.window_stride,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if norm_params:
            metadata.update(norm_params)
            
        metadata_file = os.path.join(self.output_dir, f"{filename_base}_metadata.csv")
        pd.DataFrame([metadata]).to_csv(metadata_file, index=False)
        self.logger.info(f"Metadata saved to {metadata_file}")
    
    def _save_separate_data(self, data, processed_appliances, norm_params):
        """保存分离模式数据（每个电器一个文件）"""
        # 保存每个电器的数据
        aggregate_column = self.aggregate_name
        
        for appliance in tqdm(processed_appliances, desc="Saving individual appliance files"):
            # 提取单个电器的数据
            columns_to_include = [aggregate_column, appliance]
            status_col = f"{appliance}_status"
            if status_col in data.columns:
                columns_to_include.append(status_col)
                
            appliance_data = data[columns_to_include]
            
            # 生成文件名
            filename_base = f"{self.dataset}_{appliance}"
            
            # 检查是否存在已处理的文件
            output_file = os.path.join(self.output_dir, f"{filename_base}.csv")
            if self.skip_existing and os.path.exists(output_file):
                self.logger.info(f"File {output_file} already exists, skipping...")
                continue
            
            # 保存数据
            compression = 'gzip' if self.compress else None
            extension = '.csv.gz' if self.compress else '.csv'
            
            output_file = os.path.join(self.output_dir, f"{filename_base}{extension}")
            appliance_data.to_csv(output_file, compression=compression)
            self.logger.info(f"Data for {appliance} saved to {output_file}")
            
            # 保存状态信息
            if status_col in appliance_data.columns:
                status_file = os.path.join(self.output_dir, f"{filename_base}_status{extension}")
                status_data = appliance_data[[status_col]]
                status_data.to_csv(status_file, compression=compression)
                self.logger.info(f"Status data for {appliance} saved to {status_file}")
            
            # 保存标准化参数
            if norm_params:
                stats_file = os.path.join(self.output_dir, f"{filename_base}_stats.csv")
                pd.DataFrame([norm_params]).to_csv(stats_file, index=False)
                self.logger.info(f"Normalization stats for {appliance} saved to {stats_file}")
            
            # 保存数据信息
            metadata = {
                'dataset': self.dataset,
                'method': self.method,
                'houses': self.houses,
                'appliance': appliance,
                'total_samples': len(appliance_data),
                'sampling_rate': self.sampling,
                'normalization': self.normalize,
                'window_size': self.window_size,
                'window_stride': self.window_stride,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            if norm_params:
                metadata.update(norm_params)
                
            metadata_file = os.path.join(self.output_dir, f"{filename_base}_metadata.csv")
            pd.DataFrame([metadata]).to_csv(metadata_file, index=False)
            self.logger.info(f"Metadata for {appliance} saved to {metadata_file}")

def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建处理器并处理数据
    processor = NILMDataProcessor(args)
    processor.process_dataset()

if __name__ == "__main__":
    main()