```

(14-LLaMA-Factory) root@anhua210:/Data/CODE/14-LLaMA-Factory/LLaMA-Factory# pip list
Package                  Version
------------------------ -----------
accelerate               1.11.0
aiofiles                 24.1.0
aiohappyeyeballs         2.6.1
aiohttp                  3.13.2
aiosignal                1.4.0
annotated-doc            0.0.4
annotated-types          0.7.0
antlr4-python3-runtime   4.9.3
anyio                    4.12.0
attrs                    25.4.0
audioread                3.1.0
av                       16.0.1
brotli                   1.2.0
certifi                  2025.11.12
cffi                     2.0.0
charset-normalizer       3.4.4
click                    8.3.1
contourpy                1.3.3
cycler                   0.12.1
datasets                 4.0.0
decorator                5.2.1
dill                     0.3.8
docstring_parser         0.17.0
einops                   0.8.1
fastapi                  0.124.0
ffmpy                    1.0.0
filelock                 3.20.0
fire                     0.7.1
fonttools                4.61.0
frozenlist               1.8.0
fsspec                   2025.3.0
gradio                   5.45.0
gradio_client            1.13.0
groovy                   0.1.2
h11                      0.16.0
hf_transfer              0.1.9
hf-xet                   1.2.0
httpcore                 1.0.9
httpx                    0.28.1
huggingface-hub          0.36.0
idna                     3.11
Jinja2                   3.1.6
joblib                   1.5.2
kiwisolver               1.4.9
lazy_loader              0.4
librosa                  0.11.0
llvmlite                 0.45.1
markdown-it-py           4.0.0
MarkupSafe               3.0.3
matplotlib               3.10.7
mdurl                    0.1.2
modelscope               1.32.0
mpmath                   1.3.0
msgpack                  1.1.2
multidict                6.7.0
multiprocess             0.70.16
networkx                 3.6
numba                    0.62.1
numpy                    1.26.4
nvidia-cublas-cu12       12.8.4.1
nvidia-cuda-cupti-cu12   12.8.90
nvidia-cuda-nvrtc-cu12   12.8.93
nvidia-cuda-runtime-cu12 12.8.90
nvidia-cudnn-cu12        9.10.2.21
nvidia-cufft-cu12        11.3.3.83
nvidia-cufile-cu12       1.13.1.3
nvidia-curand-cu12       10.3.9.90
nvidia-cusolver-cu12     11.7.3.90
nvidia-cusparse-cu12     12.5.8.93
nvidia-cusparselt-cu12   0.7.1
nvidia-nccl-cu12         2.27.5
nvidia-nvjitlink-cu12    12.8.93
nvidia-nvshmem-cu12      3.3.20
nvidia-nvtx-cu12         12.8.90
omegaconf                2.3.0
orjson                   3.11.5
packaging                25.0
pandas                   2.3.3
peft                     0.17.1
pillow                   11.3.0
pip                      24.0
platformdirs             4.5.1
pooch                    1.8.2
propcache                0.4.1
protobuf                 6.33.2
psutil                   7.1.3
pyarrow                  22.0.0
pycparser                2.23
pydantic                 2.10.6
pydantic_core            2.27.2
pydub                    0.25.1
Pygments                 2.19.2
pyparsing                3.2.5
python-dateutil          2.9.0.post0
python-multipart         0.0.20
pytz                     2025.2
PyYAML                   6.0.3
regex                    2025.11.3
requests                 2.32.5
rich                     14.2.0
ruff                     0.14.8
safehttpx                0.1.7
safetensors              0.5.3
scikit-learn             1.7.2
scipy                    1.16.3
semantic-version         2.10.0
sentencepiece            0.2.1
setuptools               80.9.0
shellingham              1.5.4
shtab                    1.8.0
six                      1.17.0
soundfile                0.13.1
soxr                     1.0.0
sse-starlette            3.0.3
starlette                0.50.0
sympy                    1.14.0
termcolor                3.2.0
threadpoolctl            3.6.0
tiktoken                 0.12.0
tokenizers               0.22.1
tomlkit                  0.13.3
torch                    2.9.1
tqdm                     4.67.1
transformers             4.57.1
triton                   3.5.1
trl                      0.9.6
typer                    0.20.0
typing_extensions        4.15.0
tyro                     0.8.14
tzdata                   2025.2
urllib3                  2.6.0
uvicorn                  0.38.0
websockets               15.0.1
xxhash                   3.6.0
yarl                     1.22.0




(14-LLaMA-Factory) root@anhua210:/Data/CODE/14-LLaMA-Factory/LLaMA-Factory# cat whl_url.txt |grep http |grep Obtaining
  Obtaining dependency information for transformers!=4.52.0,!=4.57.0,<=4.57.1,>=4.49.0 from https://files.pythonhosted.org/packages/71/d3/c16c3b3cf7655a67db1144da94b021c200ac1303f82428f2beef6c2e72bb/transformers-4.57.1-py3-none-any.whl.metadata
  Obtaining dependency information for accelerate<=1.11.0,>=1.3.0 from https://files.pythonhosted.org/packages/77/85/85951bc0f9843e2c10baaa1b6657227056095de08f4d1eea7d8b423a6832/accelerate-1.11.0-py3-none-any.whl.metadata
  Obtaining dependency information for trl<=0.9.6,>=0.8.6 from https://files.pythonhosted.org/packages/a5/c3/6565c2c376a829f99da20d39c2912405195ec1fa6aae068dc45c46793e72/trl-0.9.6-py3-none-any.whl.metadata
  Obtaining dependency information for matplotlib>=3.7.0 from https://files.pythonhosted.org/packages/7d/18/95ae2e242d4a5c98bd6e90e36e128d71cf1c7e39b0874feaed3ef782e789/matplotlib-3.10.7-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata
  Obtaining dependency information for tyro<0.9.0 from https://files.pythonhosted.org/packages/60/ec/e34d546cfd9c5b906d1d534bb75557be9f2b179609d60bb9e97ec07e8ead/tyro-0.8.14-py3-none-any.whl.metadata
  Obtaining dependency information for numpy<2.0.0 from https://files.pythonhosted.org/packages/0f/50/de23fde84e45f5c4fda2488c759b69990fd4512387a8632860f3ac9cd225/numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata
  Obtaining dependency information for pandas>=2.0.0 from https://files.pythonhosted.org/packages/e5/63/cd7d615331b328e287d8233ba9fdf191a9c2d11b6af0c7a59cfcec23de68/pandas-2.3.3-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl.metadata
  Obtaining dependency information for scipy from https://files.pythonhosted.org/packages/79/2e/415119c9ab3e62249e18c2b082c07aff907a273741b3f8160414b0e9193c/scipy-1.16.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata
  Obtaining dependency information for sentencepiece from https://files.pythonhosted.org/packages/04/88/14f2f4a2b922d8b39be45bf63d79e6cd3a9b2f248b2fcb98a69b12af12f5/sentencepiece-0.2.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata
  Obtaining dependency information for tiktoken from https://files.pythonhosted.org/packages/f4/90/3dae6cc5436137ebd38944d396b5849e167896fc2073da643a49f372dc4f/tiktoken-0.12.0-cp312-cp312-manylinux_2_28_x86_64.whl.metadata
  Obtaining dependency information for modelscope>=1.14.0 from https://files.pythonhosted.org/packages/64/92/b24fd3d91d87bf2189a422d6acea736505616b54706eadf141d2298c7a1b/modelscope-1.32.0-py3-none-any.whl.metadata
  Obtaining dependency information for hf-transfer from https://files.pythonhosted.org/packages/d6/d8/f87ea6f42456254b48915970ed98e993110521e9263472840174d32c880d/hf_transfer-0.1.9-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata
  Obtaining dependency information for safetensors<=0.5.3 from https://files.pythonhosted.org/packages/a6/f8/dae3421624fcc87a89d42e1898a798bc7ff72c61f38973a65d60df8f124c/safetensors-0.5.3-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata
  Obtaining dependency information for fire from https://files.pythonhosted.org/packages/e5/4c/93d0f85318da65923e4b91c1c2ff03d8a458cbefebe3bc612a6693c7906d/fire-0.7.1-py3-none-any.whl.metadata
  Obtaining dependency information for omegaconf from https://files.pythonhosted.org/packages/e3/94/1843518e420fa3ed6919835845df698c7e27e183cb997394e4a670973a65/omegaconf-2.3.0-py3-none-any.whl.metadata
  Obtaining dependency information for protobuf from https://files.pythonhosted.org/packages/56/13/333b8f421738f149d4fe5e49553bc2a2ab75235486259f689b4b91f96cec/protobuf-6.33.2-cp39-abi3-manylinux2014_x86_64.whl.metadata
  Obtaining dependency information for pyyaml from https://files.pythonhosted.org/packages/8b/9d/b3589d3877982d4f2329302ef98a8026e7f4443c765c46cfecc8858c6b4b/pyyaml-6.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata
  Obtaining dependency information for pydantic<=2.10.6 from https://files.pythonhosted.org/packages/f4/3c/8cc1cc84deffa6e25d2d0c688ebb80635dfdbf1dbea3e30c541c8cf4d860/pydantic-2.10.6-py3-none-any.whl.metadata
  Obtaining dependency information for fastapi from https://files.pythonhosted.org/packages/4d/29/9e1e82e16e9a1763d3b55bfbe9b2fa39d7175a1fd97685c482fa402e111d/fastapi-0.124.0-py3-none-any.whl.metadata
  Obtaining dependency information for sse-starlette from https://files.pythonhosted.org/packages/23/a0/984525d19ca5c8a6c33911a0c164b11490dd0f90ff7fd689f704f84e9a11/sse_starlette-3.0.3-py3-none-any.whl.metadata
  Obtaining dependency information for av from https://files.pythonhosted.org/packages/b2/7a/1305243ab47f724fdd99ddef7309a594e669af7f0e655e11bdd2c325dfae/av-16.0.1-cp312-cp312-manylinux_2_28_x86_64.whl.metadata
  Obtaining dependency information for propcache!=0.4.0 from https://files.pythonhosted.org/packages/46/4b/3aae6835b8e5f44ea6a68348ad90f78134047b503765087be2f9912140ea/propcache-0.4.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata
  Obtaining dependency information for regex!=2019.12.17 from https://files.pythonhosted.org/packages/84/bd/9ce9f629fcb714ffc2c3faf62b6766ecb7a585e1e885eb699bcf130a5209/regex-2025.11.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata
  Obtaining dependency information for tokenizers<=0.23.0,>=0.22.0 from https://files.pythonhosted.org/packages/d0/c6/dc3a0db5a6766416c32c034286d7c2d406da1f498e4de04ab1b8959edd00/tokenizers-0.22.1-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata
  Obtaining dependency information for pyarrow>=15.0.0 from https://files.pythonhosted.org/packages/13/95/aec81f781c75cd10554dc17a25849c720d54feafb6f7847690478dcf5ef8/pyarrow-22.0.0-cp312-cp312-manylinux_2_28_x86_64.whl.metadata
  Obtaining dependency information for xxhash from https://files.pythonhosted.org/packages/11/4f/426f91b96701ec2f37bb2b8cec664eff4f658a11f3fa9d94f0a887ea6d2b/xxhash-3.6.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata
  Obtaining dependency information for multiprocess<0.70.17 from https://files.pythonhosted.org/packages/0a/7d/a988f258104dcd2ccf1ed40fdc97e26c4ac351eeaf81d76e266c52d84e2f/multiprocess-0.70.16-py312-none-any.whl.metadata
  Obtaining dependency information for psutil from https://files.pythonhosted.org/packages/ce/b1/5f49af514f76431ba4eea935b8ad3725cdeb397e9245ab919dbc1d1dc20f/psutil-7.1.3-cp36-abi3-manylinux2010_x86_64.manylinux_2_12_x86_64.manylinux_2_28_x86_64.whl.metadata
  Obtaining dependency information for torch>=2.0.0 from https://files.pythonhosted.org/packages/19/17/e377a460603132b00760511299fceba4102bd95db1a0ee788da21298ccff/torch-2.9.1-cp312-cp312-manylinux_2_28_x86_64.whl.metadata
  Obtaining dependency information for anyio<5.0,>=3.0 from https://files.pythonhosted.org/packages/7f/9c/36c5c37947ebfb8c7f22e0eb6e4d188ee2d53aa3880f3f2744fb894f0cb1/anyio-4.12.0-py3-none-any.whl.metadata
  Obtaining dependency information for brotli>=1.1.0 from https://files.pythonhosted.org/packages/03/a7/03aa61fbc3c5cbf99b44d158665f9b0dd3d8059be16c460208d9e385c837/brotli-1.2.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata
  Obtaining dependency information for markupsafe<4.0,>=2.0 from https://files.pythonhosted.org/packages/3c/2e/8d0c2ab90a8c1d9a24f0399058ab8519a3279d1bd4289511d74e909f060e/markupsafe-3.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata
  Obtaining dependency information for orjson~=3.0 from https://files.pythonhosted.org/packages/46/bf/0993b5a056759ba65145effe3a79dd5a939d4a070eaa5da2ee3180fbb13f/orjson-3.11.5-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata
  Obtaining dependency information for pillow<12.0,>=8.0 from https://files.pythonhosted.org/packages/e4/c9/06dd4a38974e24f932ff5f98ea3c546ce3f8c995d3f0985f8e5ba48bba19/pillow-11.3.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata
  Obtaining dependency information for ruff>=0.9.3 from https://files.pythonhosted.org/packages/96/bc/058fe0aefc0fbf0d19614cb6d1a3e2c048f7dc77ca64957f33b12cfdc5ef/ruff-0.14.8-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata
  Obtaining dependency information for websockets<16.0,>=10.0 from https://files.pythonhosted.org/packages/14/8f/aa61f528fba38578ec553c145857a181384c72b98156f858ca5c8e82d9d3/websockets-15.0.1-cp312-cp312-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata
  Obtaining dependency information for contourpy>=1.0.1 from https://files.pythonhosted.org/packages/cc/8f/ec6289987824b29529d0dfda0d74a07cec60e54b9c92f3c9da4c0ac732de/contourpy-1.3.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata
  Obtaining dependency information for fonttools>=4.22.0 from https://files.pythonhosted.org/packages/29/a3/1fa90b95b690f0d7541f48850adc40e9019374d896c1b8148d15012b2458/fonttools-4.61.0-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl.metadata
  Obtaining dependency information for kiwisolver>=1.3.1 from https://files.pythonhosted.org/packages/70/90/6d240beb0f24b74371762873e9b7f499f1e02166a2d9c5801f4dbf8fa12e/kiwisolver-1.4.9-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata
  Obtaining dependency information for docstring-parser>=0.16 from https://files.pythonhosted.org/packages/55/e2/2537ebcff11c1ee1ff17d8d0b6f4db75873e3b0fb32c2d4a2ee31ecb310a/docstring_parser-0.17.0-py3-none-any.whl.metadata
  Obtaining dependency information for shtab>=1.5.6 from https://files.pythonhosted.org/packages/8e/e1/202a31727b0d096a04380f78e809074d7a1d0a22d9d5a39fea1d2353fd02/shtab-1.8.0-py3-none-any.whl.metadata
  Obtaining dependency information for urllib3>=1.26 from https://files.pythonhosted.org/packages/56/1a/9ffe814d317c5224166b23e7c47f606d6e473712a2fad0f704ea9b99f246/urllib3-2.6.0-py3-none-any.whl.metadata
  Obtaining dependency information for termcolor from https://files.pythonhosted.org/packages/f9/d5/141f53d7c1eb2a80e6d3e9a390228c3222c27705cbe7f048d3623053f3ca/termcolor-3.2.0-py3-none-any.whl.metadata
  Obtaining dependency information for pydantic-core==2.27.2 from https://files.pythonhosted.org/packages/8d/f0/49129b27c43396581a635d8710dae54a791b17dfc50c70164866bbf865e3/pydantic_core-2.27.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata
  Obtaining dependency information for numba>=0.51.0 from https://files.pythonhosted.org/packages/50/5f/6a802741176c93f2ebe97ad90751894c7b0c922b52ba99a4395e79492205/numba-0.62.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata
  Obtaining dependency information for scikit-learn>=1.1.0 from https://files.pythonhosted.org/packages/5c/d0/0c577d9325b05594fdd33aa970bf53fb673f051a45496842caee13cfd7fe/scikit_learn-1.7.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata
  Obtaining dependency information for soxr>=0.3.2 from https://files.pythonhosted.org/packages/b1/87/2726603c13c2126cb8ded9e57381b7377f4f0df6ba4408e1af5ddbfdc3dd/soxr-1.0.0-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata
  Obtaining dependency information for msgpack>=1.0 from https://files.pythonhosted.org/packages/65/92/a5100f7185a800a5d29f8d14041f61475b9de465ffcc0f3b9fba606e4505/msgpack-1.1.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata
  Obtaining dependency information for aiohttp!=4.0.0a0,!=4.0.0a1 from https://files.pythonhosted.org/packages/b9/48/adf56e05f81eac31edcfae45c90928f4ad50ef2e3ea72cb8376162a368f8/aiohttp-3.13.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata
  Obtaining dependency information for llvmlite<0.46,>=0.45.0dev0 from https://files.pythonhosted.org/packages/96/76/0f7154952f037cb320b83e1c952ec4a19d5d689cf7d27cb8a26887d7bbc1/llvmlite-0.45.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata
  Obtaining dependency information for platformdirs>=2.5.0 from https://files.pythonhosted.org/packages/cb/28/3bfe2fa5a7b9c46fe7e13c97bda14c895fb10fa2ebf1d0abb90e0cea7ee1/platformdirs-4.5.1-py3-none-any.whl.metadata
  Obtaining dependency information for charset_normalizer<4,>=2 from https://files.pythonhosted.org/packages/c0/10/d20b513afe03acc89ec33948320a5544d31f21b05368436d580dec4e234d/charset_normalizer-3.4.4-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata
  Obtaining dependency information for cffi>=1.0 from https://files.pythonhosted.org/packages/78/2d/7fa73dfa841b5ac06c7b8855cfc18622132e365f5b81d02230333ff26e9e/cffi-2.0.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata
  Obtaining dependency information for networkx>=2.5.1 from https://files.pythonhosted.org/packages/07/c7/d64168da60332c17d24c0d2f08bdf3987e8d1ae9d84b5bbd0eec2eb26a55/networkx-3.6-py3-none-any.whl.metadata
  Obtaining dependency information for triton==3.5.1 from https://files.pythonhosted.org/packages/f2/50/9a8358d3ef58162c0a415d173cfb45b67de60176e1024f71fbc4d24c0b6d/triton-3.5.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata
  Obtaining dependency information for frozenlist>=1.1.1 from https://files.pythonhosted.org/packages/6a/bd/d91c5e39f490a49df14320f4e8c80161cfcce09f1e2cde1edd16a551abb3/frozenlist-1.8.0-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl.metadata
  Obtaining dependency information for multidict<7.0,>=4.5 from https://files.pythonhosted.org/packages/0d/e2/9baffdae21a76f77ef8447f1a05a96ec4bc0a24dae08767abc0a2fe680b8/multidict-6.7.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata
  Obtaining dependency information for yarl<2.0,>=1.17.0 from https://files.pythonhosted.org/packages/db/0f/0d52c98b8a885aeda831224b78f3be7ec2e1aa4a62091f9f9188c3c65b56/yarl-1.22.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata

```
