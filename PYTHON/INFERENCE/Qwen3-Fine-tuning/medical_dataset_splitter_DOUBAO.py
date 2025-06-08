from modelscope.msdatasets import MsDataset
import json
import random
import os
from tqdm import tqdm
import argparse

class MedicalDatasetSplitter:
    """医疗数据集分割工具，支持按比例分割数据集并保存为jsonl格式"""
    
    def __init__(self, seed=42):
        """初始化分割器，设置随机种子以确保结果可复现"""
        self.seed = seed
        random.seed(seed)
        print(f"随机种子已设置为{seed}，确保分割结果可复现")
    
    def load_dataset(self, dataset_name, subset_name='default', split='train'):
        """从ModelScope加载数据集"""
        print(f"正在加载数据集: {dataset_name} (子集: {subset_name}, 分割: {split})")
        try:
            dataset = MsDataset.load(dataset_name, subset_name=subset_name, split=split)
            print(f"数据集加载成功，共{len(dataset)}条数据")
            return dataset
        except Exception as e:
            print(f"数据集加载失败: {e}")
            return None
    
    def split_dataset(self, dataset, train_ratio=0.9):
        """将数据集按比例分割为训练集和验证集"""
        if not dataset:
            raise ValueError("数据集为空，无法进行分割")
            
        data_list = list(dataset)
        print(f"正在打乱并分割数据集，训练集比例: {train_ratio}")
        
        # 打乱数据集
        random.shuffle(data_list)
        
        # 计算分割索引
        split_idx = int(len(data_list) * train_ratio)
        
        # 分割数据集
        train_data = data_list[:split_idx]
        val_data = data_list[split_idx:]
        
        print(f"数据集分割完成:")
        print(f"  训练集大小: {len(train_data)}")
        print(f"  验证集大小: {len(val_data)}")
        
        return train_data, val_data
    
    def save_to_jsonl(self, data, file_path, encoding='utf-8'):
        """将数据保存为jsonl格式"""
        if not data:
            print(f"数据为空，跳过保存文件: {file_path}")
            return
            
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        print(f"正在将数据保存到: {file_path}")
        with open(file_path, 'w', encoding=encoding) as f:
            for item in tqdm(data, desc="保存进度"):
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        print(f"文件保存成功，共{len(data)}条记录")
    
    def run(self, dataset_name, output_dir='./', train_ratio=0.9, 
            subset_name='default', split='train'):
        """执行完整的数据集分割流程"""
        # 加载数据集
        dataset = self.load_dataset(dataset_name, subset_name, split)
        
        # 分割数据集
        train_data, val_data = self.split_dataset(dataset, train_ratio)
        
        # 保存分割结果
        train_file = os.path.join(output_dir, 'train.jsonl')
        val_file = os.path.join(output_dir, 'val.jsonl')
        
        self.save_to_jsonl(train_data, train_file)
        self.save_to_jsonl(val_data, val_file)
        
        print(f"\n数据集分割完成!")
        print(f"训练集路径: {train_file}")
        print(f"验证集路径: {val_file}")

def main():
    parser = argparse.ArgumentParser(description='医疗数据集分割工具')
    parser.add_argument('--dataset_name', type=str, required=True, 
                        help='ModelScope上的数据集名称')
    parser.add_argument('--output_dir', type=str, default='./', 
                        help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.9, 
                        help='训练集比例，范围(0,1)')
    parser.add_argument('--subset_name', type=str, default='default', 
                        help='数据集子集名称')
    parser.add_argument('--split', type=str, default='train', 
                        help='要加载的数据集分割')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 创建分割器实例并执行分割
    splitter = MedicalDatasetSplitter(seed=args.seed)
    splitter.run(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        subset_name=args.subset_name,
        split=args.split
    )

if __name__ == "__main__":
    main()
