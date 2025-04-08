### Boltz_Partial_Diffusion_I_Subquotient_C3

An important note is that partial_diffusion_setup.py will create 2 triangles. Ensure that it creates [A,B,C] matches up with [D,E,F]. Importantly, when using the I subquotient C3 matrices (size=20), ensure that A maps to D, B maps to E, and D maps to F. The code should be doing this right now, but just verify using a few examples, as it has the potential to break down the line functionality. 


### File Replacements:

Note that these file paths are based on the Filezilla boltz env folder. I believe they are general, but am not completely sure. 

#### main.py
Replace: /work/keshavsundar/env/boltz_glycan/lib/python3.10/site-packages/boltz/main.py

#### model.py
Replace: /work/keshavsundar/env/boltz_glycan/lib/python3.10/site-packages/boltz/model/model.py

#### diffusion.py
Replace: /work/keshavsundar/env/boltz_glycan/lib/python3.10/site-packages/boltz/model/modules/diffusion.py

#### symmetry.py
Add: /work/keshavsundar/env/boltz_glycan/lib/python3.10/site-packages/boltz/data/module/symmetry_awareness.py
