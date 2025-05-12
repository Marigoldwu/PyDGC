# -*- coding: utf-8 -*-
from .base_pipeline import BasePipeline
from .gae_pipeline import GAEPipeline
from .gae_ssc_pipeline import GAESSCPipeline
from .daegc_pipeline import DAEGCPipeline
from .sdcn_pipeline import SDCNPipeline
from .dfcn_pipeline import DFCNPipeline
from .dcrn_pipeline import DCRNPipeline
from .agcdrr_pipeline import AGCDRRPipeline
from .hsan_pipeline import HSANPipeline
from .ccgc_pipeline import CCGCPipeline
from .dgcluster_pipeline import DGCLUSTERPipeline
from .magi_pipeline import MAGIPipeline
from .ns4gc_pipeline import NS4GCPipeline


__all__ = [
    'BasePipeline',
    'GAEPipeline',
    'GAESSCPipeline',
    'DAEGCPipeline',
    'SDCNPipeline',
    'DFCNPipeline',
    'DCRNPipeline',
    'AGCDRRPipeline',
    'HSANPipeline',
    'CCGCPipeline',
    'DGCLUSTERPipeline',
    'MAGIPipeline',
    'NS4GCPipeline'
]
