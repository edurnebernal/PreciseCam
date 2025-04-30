# —————  create env ——————
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia --yes

git clone https://github.com/edurnebernal/diffusers-adapted.git
conda install -c conda-forge diffusers --yes

pip install -e ./diffusers-adapted
pip install -r ./diffusers-adapted/examples/controlnet/requirements_sdxl.txt
# pip install accelerate==0.31.0
accelerate config default
pip install -U scikit-learn
pip install pyequilib==0.3.0
pip install gradio
pip install opencv-python
pip install matplotlib==3.9.2