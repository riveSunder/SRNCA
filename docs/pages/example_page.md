# Example of another page

These equations describe the Gray-Scott reaction diffusion model

$$
\frac{\partial u}{\partial t} = r_u \nabla^2 u - uv^2 + f(1-u)
$$

$$
\frac{\partial v}{\partial t} = r_v \nabla^2 v + uv^2 - v(f+k)
$$

However they don't render in Jekyll's Kramdown. Instead we can use `sympy`'s preview method to generate equation images and then include these in pages as normal:


{:style="text-align:center;"}
[![Gray-Scott equation for change in u](https://github.com/riveSunder/SRNCA/blob/master/assets/gray_scott_dudt.png?raw=true)](https://github.com/riveSunder/SRNCA/blob/master/assets/gray_scott_dudt.png?raw=true)

{:style="text-align:center;"}
[![Gray-Scott equation for change in v](https://github.com/riveSunder/SRNCA/blob/master/assets/gray_scott_dvdt.png?raw=true)](https://github.com/riveSunder/SRNCA/blob/master/assets/gray_scott_dvdt.png?raw=true)

There is an [example notebook](https://github.com/riveSunder/SRNCA/blob/master/notebooks/render_math.ipynb) which you can try in [mybinder](https://mybinder.org/v2/gh/rivesunder/srnca/master?labpath=notebooks%2Frender_math.ipynb) -> [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rivesunder/srnca/master?labpath=notebooks%2Frender_math.ipynb) or in [colab](https://colab.research.google.com/github/rivesunder/srnca/blob/master/notebooks/render_math.ipynb) -> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rivesunder/srnca/blob/master/notebooks/render_math.ipynb) 
