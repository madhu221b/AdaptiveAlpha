import os
import re

dirname = "./AdaptiveAlpha/fastadaptivealphatestfixed_alpha_0.5_beta_2.0_fm_0.3"
pattern = re.compile(r"^((?!29).)*$")

for filename in os.listdir(dirname):
    print(filename)
    if pattern.match(filename):
        try:
            os.remove(os.path.join(dirname, filename))
        except EnvironmentError:
            pass
