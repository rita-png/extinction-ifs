import numpy as np
import matplotlib.pyplot as plt


ew_val=0.99

na_rest=(5890+5896)/2

SN_name="SN2010ev"
if SN_name=="SN2010ev":
    z=0.00921
    ra, dec = 156.370792, -39.830889


w,f = np.load("DATA/"+SN_name+"/outliers/"+str(ew_val)+".npy")

plt.plot(w,f)
plt.xlim(na_rest-80,na_rest+80)




plt.savefig("quickplots/"+str(ew_val)+".pdf", bbox_inches='tight')