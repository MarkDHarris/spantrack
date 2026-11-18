uv venv --python 3.12
source .venv/bin/activate

uv sync --all-groups

python -m pip install "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"


python -c "import tqdm; print('tqdm OK')"

python -m pip install "tqdm>=4.66,<5"

python -m pip install -e ".[progress,dev]"



uv run spantrack ./your_book.epub


spantrack ~/Downloads/test/How_Large_Language_Models_Work_fixed.epub --voice af_bella --bitrate 64k --progress

spantrack ~/Downloads/test/How_Large_Language_Models_Work.epub --voice af_heart --bitrate 64k --progress








spantrack ~/Downloads/test/How_Large_Language_Models_Work.epub --voice af_heart --bitrate 64k --progress
narration set: 19 document(s)
  narrate  OEBPS/Text/title.html
  narrate  OEBPS/Text/copyright.html
  narrate  OEBPS/Text/contents.html
  narrate  OEBPS/Text/preface.html
  narrate  OEBPS/Text/acknowledgments.html
  narrate  OEBPS/Text/about-this-book.html
  narrate  OEBPS/Text/about-the-authors.html
  narrate  OEBPS/Text/about-the-cover-illustration.html
  narrate  OEBPS/Text/chapter-1.html
  narrate  OEBPS/Text/chapter-2.html
  narrate  OEBPS/Text/chapter-3.html
  narrate  OEBPS/Text/chapter-4.html
  narrate  OEBPS/Text/chapter-5.html
  narrate  OEBPS/Text/chapter-6.html
  narrate  OEBPS/Text/chapter-7.html
  narrate  OEBPS/Text/chapter-8.html
  narrate  OEBPS/Text/chapter-9.html
  narrate  OEBPS/Text/references.html
  narrate  OEBPS/Text/index.html
  skip     OEBPS/Text/titlepage.xhtml (landmark:cover)
projected: 420,307 characters, ~6h 59m 28s audio, ~192.0 MB
W1118 09:27:39.582000 29391 torch/distributed/elastic/multiprocessing/redirects.py:35] NOTE: Redirects are currently not supported in MacOs.
/Users/markharris/Code/spantrack/.venv/lib/python3.12/site-packages/torch/nn/modules/rnn.py:1012: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.2 and num_layers=1
  super().__init__("LSTM", *args, **kwargs)
/Users/markharris/Code/spantrack/.venv/lib/python3.12/site-packages/torch/nn/utils/weight_norm.py:145: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)
/Users/markharris/Code/spantrack/.venv/lib/python3.12/site-packages/torch/jit/_script.py:1491: FutureWarning: `torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.
  warnings.warn(
narrating:  11%|██████████




These four warnings come from PyTorch and the upstream neural network architecture defined inside the Kokoro TTS model weights. They are advisory and do not indicate corrupted audio or failed processing.

1. torch/distributed/elastic/multiprocessing/redirects.py: NOTE: Redirects are currently not supported in MacOs.
What it means: PyTorch’s distributed process launcher includes standard utilities to capture and redirect child process standard I/O (stdout/stderr). On macOS, this platform hook is disabled because POSIX file descriptor redirection behaves differently than on Linux.

Impact: None. Kokoro is running locally on your Apple Silicon hardware via standard single-node MPS/CPU inference; it does not rely on distributed elastic training redirects.

2. torch/nn/modules/rnn.py: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.2 and num_layers=1
What it means: In Kokoro's text encoder / duration predictor architecture, an LSTM recurrent layer was initialized with num_layers=1 and dropout=0.2.

Technical context: In PyTorch, the dropout parameter on an nn.LSTM module only applies to the connections between stacked recurrent layers. When there is only a single layer (num_layers=1), PyTorch drops no activations, rendering the dropout=0.2 setting a no-op.

Impact: None during inference. Dropout is strictly a training-time regularization technique that is inactive during evaluation (model.eval()).

3. torch/nn/utils/weight_norm.py: FutureWarning: torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.
What it means: The audio decoder/vocoder in Kokoro (derived from HiFi-GAN / StyleTTS2 architectures) uses weight normalization on its convolutional layers.

Technical context: The legacy torch.nn.utils.weight_norm hook modifies module state via older PyTorch hooks. PyTorch introduced a cleaner, unified parameterization API (torch.nn.utils.parametrizations.weight_norm) and marked the legacy hook for eventual removal in future major versions.

Impact: None in PyTorch 2.x. It is a backwards-compatibility deprecation notice informing library maintainers to update the call before PyTorch 3.0.

4. torch/jit/_script.py: FutureWarning: torch.jit.script is deprecated. Please switch to torch.compile or torch.export.
What it means: When loading the voice model or backend kernels, Kokoro calls torch.jit.script(...) to compile small mathematical helper routines into TorchScript bytecode.

Technical context: PyTorch 2.0+ emphasizes torch.compile() (TorchDynamo / Inductor) and torch.export over TorchScript (JIT script/trace).

Impact: None. TorchScript remains fully operational throughout PyTorch 2.x.

Suppressing the Output (Optional)
If you prefer to keep the CLI and progress bars clean without PyTorch deprecation notices filling the terminal buffer, you can mute standard warnings by running with the PYTHONWARNINGS environment variable:

Bash
PYTHONWARNINGS="ignore" spantrack ~/Downloads/test/How_Large_Language_Models_Work_fixed.epub --voice af_heart --bitrate 64k --progress


Or pass Python's -W ignore flag directly:

Bash
python -W ignore -m spantrack.cli ~/Downloads/test/How_Large_Language_Models_Work_fixed.epub --voice af_heart --bitrate 64k --progress


