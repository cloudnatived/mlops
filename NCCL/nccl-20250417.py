#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# Copyright (c) PLUMgrid, Inc.
# Licensed under the Apache License, Version 2.0 (the "License")
# 22-Apr-2025

import ctypes
so_file = "/lib/x86_64-linux-gnu/libnccl.so.2"
nccl = ctypes.CDLL(so_file)
ncclResult_t = ctypes.c_int
class NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]
_c_ncclGetUniqueId = nccl.ncclGetUniqueId
_c_ncclGetUniqueId.restype = ctypes.c_int
_c_ncclGetUniqueId.argtypes = [ctypes.POINTER(NcclUniqueId)]
def ncclGetUniqueId() -> NcclUniqueId:
    unique_id = NcclUniqueId()
    result = _c_ncclGetUniqueId(ctypes.byref(unique_id))
    assert result == 0
    return unique_id

unique_id = ncclGetUniqueId()
print(list(unique_id.internal))
