# Super Resolution:
##### (done using transfer learning in pytorch)
The Super-Resolution Generative Adversarial Network (SRGAN) is a seminal work that is capable of generating realistic textures during single image super-resolution. However, the hallucinated details are often accompanied with unpleasant artifacts. To further enhance the visual quality n Enhanced SRGAN (ESRGAN) introduces the Residual-in-Residual Dense Block (RRDB) without batch normalization as the basic network building unit. Also it borrows the idea from relativistic GAN to let the discriminator
predict relative realness instead of the absolute value. Finally, it improves the perceptual loss by using the features before activation, which could provide stronger supervision for brightness consistency and texture recovery. Benefiting from these improvements, the proposed ESRGAN
achieves consistently better visual quality with more realistic and natural textures than SRGAN

## The Architecture of the Network:
![architecture](https://github.com/shauryabit2k18/super-resolution/blob/master/figures/architecture.jpg)

![rrdb](https://github.com/shauryabit2k18/super-resolution/blob/master/figures/RRDB.png)

## RESULTS:
![result1](https://github.com/shauryabit2k18/super-resolution/blob/master/figures/RESULT1.png)
![result2](https://github.com/shauryabit2k18/super-resolution/blob/master/figures/baboon.jpg)

## Quick Test
#### Dependencies
- Python 3
- [PyTorch >= 1.0](https://pytorch.org/) (CUDA version >= 7.5 if installing with CUDA. [More details](https://pytorch.org/get-started/previous-versions/))
- Python packages:  `pip install numpy opencv-python`

### Test models
1. Clone this github repo. 
```
git clone https://github.com/shauryabit2k18/super-resolution.git
cd super-resolution
```
2. Place your own **low-resolution images** in a new folder named as `./LR` folder.
3. Create a new folder named as 'results' in the master.
4. Run test.
```
python test.py
```
5. The results will be in a folder named `./results` folder.
