#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# Copyright (c) PLUMgrid, Inc.
# Licensed under the Apache License, Version 2.0 (the "License")

import ctypes
import sys
import os

def load_nccl_library():
    """尝试加载NCCL库"""
    # 常见的NCCL库路径
    possible_paths = [
        "/lib/x86_64-linux-gnu/libnccl.so.2",
        "/usr/lib/x86_64-linux-gnu/libnccl.so.2",
        "/usr/local/lib/libnccl.so.2",
        "/usr/lib/libnccl.so.2",
        "libnccl.so.2",  # 系统默认路径
        "libnccl.so"     # 不带版本号
    ]
    
    nccl = None
    for path in possible_paths:
        try:
            if os.path.exists(path):
                print(f"Loading NCCL from: {path}")
                nccl = ctypes.CDLL(path)
                break
        except OSError as e:
            print(f"Failed to load {path}: {e}")
            continue
    
    if nccl is None:
        # 尝试不指定路径直接加载
        try:
            nccl = ctypes.CDLL("libnccl.so.2")
        except OSError:
            try:
                nccl = ctypes.CDLL("libnccl.so")
            except OSError as e:
                print("Failed to load NCCL library")
                print("Please ensure NCCL is installed:")
                print("  sudo apt install libnccl2 libnccl-dev  # Ubuntu/Debian")
                print("  sudo yum install nccl nccl-devel      # CentOS/RHEL")
                raise e
    
    return nccl

def main():
    try:
        # 加载NCCL库
        nccl = load_nccl_library()
        
        # 定义NCCL结果类型
        ncclResult_t = ctypes.c_int
        
        # 定义ncclUniqueId结构体
        class NcclUniqueId(ctypes.Structure):
            _fields_ = [("internal", ctypes.c_byte * 128)]
        
        # 获取ncclGetUniqueId函数
        _c_ncclGetUniqueId = nccl.ncclGetUniqueId
        _c_ncclGetUniqueId.restype = ncclResult_t
        _c_ncclGetUniqueId.argtypes = [ctypes.POINTER(NcclUniqueId)]
        
        # 定义Python封装函数
        def ncclGetUniqueId() -> NcclUniqueId:
            unique_id = NcclUniqueId()
            result = _c_ncclGetUniqueId(ctypes.byref(unique_id))
            if result != 0:
                raise RuntimeError(f"ncclGetUniqueId failed with error code: {result}")
            return unique_id
        
        # 调用函数获取唯一ID
        print("Getting NCCL Unique ID...")
        unique_id = ncclGetUniqueId()
        
        # 打印结果
        print("NCCL Unique ID (first 16 bytes):")
        id_bytes = list(unique_id.internal)
        print(id_bytes[:16])  # 只打印前16字节避免输出过长
        
        print("\nFull NCCL Unique ID length:", len(id_bytes))
        print("Successfully called NCCL function!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
