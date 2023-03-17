# TW_all, TW_1 and TW_4 experiments.
## Experiments
 We show the different outputs resulting from setting some weights from the TB-model of the input channels to 0. It is trained on 15 channels. 
 
 V1: $| TW3_i | TW3_{i+26} | TW3_{i+52} | TW_{i+78} | TW_{i+104} |$

 V4: $| TW3_i | TW3_{i+26} | TW3_{i+52} | TW_{i+78} | TW_{i+104} |$

 IT: $| TW3_i | TW3_{i+26} | TW3_{i+52} | TW_{i+78} | TW_{i+104} |$

## Conditions

In this experiment, we are interested in the reconstructions resulting in a forward pass using the following 44 conditions:

---
### <b> TW_all</b>
---
<it>

- V1 + V4 + IT: $| TW3_i | TW3_{i+26} | TW3_{i+52} | TW_{i+78} | TW_{i+104} |$   
- V1: $| TW3_i | TW3_{i+26} | TW3_{i+52} | TW_{i+78} | TW_{i+104} |$   
- V4 $| TW3_i | TW3_{i+26} | TW3_{i+52} | TW_{i+78} | TW_{i+104} |$   
- IT: $| TW3_i | TW3_{i+26} | TW3_{i+52} | TW_{i+78} | TW_{i+104} |$   

---
### <b> TW_1</b>
---

All rois
- V1 + V4 + IT: $| TW3_{i} |$ 
- V1 + V4 + IT: $| TW3_{i+26} |$   
- V1 + V4 + IT: $| TW3_{i+52} |$  
- V1 + V4 + IT: $| TW3_{i+78}  |$
- V1 + V4 + IT: ß$| TW3_{i+104}  |$

---

V1
- V1: $| TW3_{i} |$
- V1: $| TW3_{i+26} |$
- V1: $| TW3_{i+52} |$
- V1: $| TW3_{i+78} |$
- V1: $| TW3_{i+104} |$

 ---

V4
- V4: $| TW3_{i} |$
- V4: $| TW3_{i+26} |$
- V4: $| TW3_{i+52} |$
- V4: $| TW3_{i+78} |$
- V4: $| TW3_{i+104} |$

---

IT
- IT: $| TW3_{i} |$
- IT: $| TW3_{i+26} |$
- IT: $| TW3_{i+52} |$
- IT: $| TW3_{i+78} |$
- IT: $| TW3_{i+104} |$

 ---
### <b> TW_4</b>
---

All rois

- V1 + V4 + IT: $| TW3_{i+26} | TW3_{i+52} | TW_{i+78} | TW_{i+104} |$ 
- V1 + V4 + IT: $| TW3_i | TW3_{i+52} | TW_{i+78} | TW_{i+104} |$   
- V1 + V4 + IT: $| TW3_i | TW3_{i+26} | TW_{i+78} | TW_{i+104} |$   
- V1 + V4 + IT: $| TW3_i | TW3_{i+26} | TW3_{i+52} | TW_{i+104} |$   
- V1 + V4 + IT: $| TW3_i | TW3_{i+26} | TW3_{i+52} | TW_{i+78} |$   

---

V1

- V1: $| TW3_{i+26} | TW3_{i+52} | TW_{i+78} | TW_{i+104} |$ 
- V1: $| TW3_i | TW3_{i+52} | TW_{i+78} | TW_{i+104} |$   
- V1: $| TW3_i | TW3_{i+26} | TW_{i+78} | TW_{i+104} |$   
- V1: $| TW3_i | TW3_{i+26} | TW3_{i+52} | TW_{i+104} |$   
- V1: $| TW3_i | TW3_{i+26} | TW3_{i+52} | TW_{i+78} |$   
 
 ---

 V4

- V4: $| TW3_{i+26} | TW3_{i+52} | TW_{i+78} | TW_{i+104} |$ 
- V4: $| TW3_i | TW3_{i+52} | TW_{i+78} | TW_{i+104} |$   
- V4: $| TW3_i | TW3_{i+26} | TW_{i+78} | TW_{i+104} |$   
- V4: $| TW3_i | TW3_{i+26} | TW3_{i+52} | TW_{i+104} |$   
- V4: $| TW3_i | TW3_{i+26} | TW3_{i+52} | TW_{i+78} |$   

---

IT

- IT: $| TW3_{i+26} | TW3_{i+52} | TW_{i+78} | TW_{i+104} |$ 
- IT: $| TW3_i | TW3_{i+52} | TW_{i+78} | TW_{i+104} |$   
- IT: $| TW3_i | TW3_{i+26} | TW_{i+78} | TW_{i+104} |$   
- IT: $| TW3_i | TW3_{i+26} | TW3_{i+52} | TW_{i+104} |$   
- IT: $| TW3_i | TW3_{i+26} | TW3_{i+52} | TW_{i+78} |$   

---

</it>

## Figures
---
### <b> TW_all</b>
---
![TW_all_all](Figures/tw_all/V1V4IT.png "TW_all_all")
![TW_all_V1](Figures/tw_all/V1.png "TW_all_V1")
![TW_all_V4](Figures/tw_all/V4.png "TW_all_V4")
![TW_all_IT](Figures/tw_all/IT.png "TW_all_IT")

---
### <b> TW_1</b>
---
#### all rois

![TW_1_all_rois_0](Figures/tw_1/all_rois/all_0.png "TW_1_all_rois_0")
![TW_1_all_rois_1](Figures/tw_1/all_rois/all_1.png "TW_1_all_rois_1")
![TW_1_all_rois_2](Figures/tw_1/all_rois/all_2.png "TW_1_all_rois_2")
![TW_1_all_rois_3](Figures/tw_1/all_rois/all_3.png "TW_1_all_rois_3")
![TW_1_all_rois_4](Figures/tw_1/all_rois/all_4.png "TW_1_all_rois_4")

---
#### V1

![TW_1_all_rois_0](Figures/tw_1/V1/V1_0.png "TW_1_IT_0")
![TW_1_all_rois_1](Figures/tw_1/V1/V1_1.png "TW_1_IT_1")
![TW_1_all_rois_2](Figures/tw_1/V1/V1_2.png "TW_1_IT_2")
![TW_1_all_rois_3](Figures/tw_1/V1/V1_3.png "TW_1_IT_3")
![TW_1_all_rois_4](Figures/tw_1/V1/V1_4.png "TW_1_IT_4")

---
#### V4

![TW_1_all_rois_0](Figures/tw_1/V4/V4_0.png "TW_1_IT_0")
![TW_1_all_rois_1](Figures/tw_1/V4/V4_1.png "TW_1_IT_1")
![TW_1_all_rois_2](Figures/tw_1/V4/V4_2.png "TW_1_IT_2")
![TW_1_all_rois_3](Figures/tw_1/V4/V4_3.png "TW_1_IT_3")
![TW_1_all_rois_4](Figures/tw_1/V4/V4_4.png "TW_1_IT_4")

---
#### IT

![TW_1_all_rois_0](Figures/tw_1/IT/IT_0.png "TW_1_IT_0")
![TW_1_all_rois_1](Figures/tw_1/IT/IT_1.png "TW_1_IT_1")
![TW_1_all_rois_2](Figures/tw_1/IT/IT_2.png "TW_1_IT_2")
![TW_1_all_rois_3](Figures/tw_1/IT/IT_3.png "TW_1_IT_3")
![TW_1_all_rois_4](Figures/tw_1/IT/IT_4.png "TW_1_IT_4")

---


---
### <b> TW_4</b>
---
#### all rois

![TW_1_all_rois_0](Figures/tw_4/all_rois/all_0.png "TW_1_all_rois_0")
![TW_1_all_rois_1](Figures/tw_4/all_rois/all_1.png "TW_1_all_rois_1")
![TW_1_all_rois_2](Figures/tw_4/all_rois/all_2.png "TW_1_all_rois_2")
![TW_1_all_rois_3](Figures/tw_4/all_rois/all_3.png "TW_1_all_rois_3")
![TW_1_all_rois_4](Figures/tw_4/all_rois/all_4.png "TW_1_all_rois_4")

---
#### V1

![TW_1_all_rois_0](Figures/tw_4/V1/V1_0.png "TW_1_IT_0")
![TW_1_all_rois_1](Figures/tw_4/V1/V1_1.png "TW_1_IT_1")
![TW_1_all_rois_2](Figures/tw_4/V1/V1_2.png "TW_1_IT_2")
![TW_1_all_rois_3](Figures/tw_4/V1/V1_3.png "TW_1_IT_3")
![TW_1_all_rois_4](Figures/tw_4/V1/V1_4.png "TW_1_IT_4")

---
#### V4

![TW_1_all_rois_0](Figures/tw_4/V4/V4_0.png "TW_1_IT_0")
![TW_1_all_rois_1](Figures/tw_4/V4/V4_1.png "TW_1_IT_1")
![TW_1_all_rois_2](Figures/tw_4/V4/V4_2.png "TW_1_IT_2")
![TW_1_all_rois_3](Figures/tw_4/V4/V4_3.png "TW_1_IT_3")
![TW_1_all_rois_4](Figures/tw_4/V4/V4_4.png "TW_1_IT_4")

---
#### IT

![TW_1_all_rois_0](Figures/tw_4/IT/IT_0.png "TW_1_IT_0")
![TW_1_all_rois_1](Figures/tw_4/IT/IT_1.png "TW_1_IT_1")
![TW_1_all_rois_2](Figures/tw_4/IT/IT_2.png "TW_1_IT_2")
![TW_1_all_rois_3](Figures/tw_4/IT/IT_3.png "TW_1_IT_3")
![TW_1_all_rois_4](Figures/tw_4/IT/IT_4.png "TW_1_IT_4")

