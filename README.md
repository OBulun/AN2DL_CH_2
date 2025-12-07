
NOTE: If possible, when a step is efficiently completed (like green splashes and Shreks removal), try to develop your Notework starting from that common point, to not have many different Notebooks completely different 

## 🎯 Currently Working On
*Update your individual status below.*

* **Karim    :** Autoaugment vs Randaugment, and initial training trials
* **Lorenzo  :**  Fine tuning a pretrained **MobileNetV3** model with a F1 score ≈ 0.4 (it's overfitting) 
* **Olgun    :** Looking into *Multi-Instance Learning -MIL-*
* **Francesco:** Adding *Image augmentation* with RandAugment, trying to use DenseNet121 (specifical for biomedical applications) to do transfer learning

---

## ⏳ Action Items (To-Do)
*List specific unassigned tasks that need to be picked up.*

- Image Augmentation
- Transfer Learning and Fine Tuning Template *if possible*


---

## 📝 Notes & Brainstorming
*Dump ideas, errors, quick thoughts, or references here.*

### Technical details
* *Insert links to documentation or resources. (if any)*

### References
* *Insert references to papers or website links from which you retrieve some models (for instance to do transfer learning) - To be inserted in the final part of the report*

---

## 🏆 Results & Findings
*Log what has been completed or discovered.*

* **Milestones:** 
    - Preprocessing -- Removal of the shrek and the green goo --

## 📘 Loogbook tips
*Things to try or already implemented (in such case put a ✅)*

* **Advices:**
- 04/12: Normalisation strategies, define the the batch size and batch norm in a reasonable way according to our specific problem
- 05/12: Inspect outleiers, we have some images that are blank after the mask is applyed (already removed), we have useless images (Shrek, already removed), we have also images wrongly labeled (How to treat them? plot the highest losses)
- 06/12: Automated augmentation, instead of applying manually just some transformations, employ strategies which learn automatically which transformations are more effective to make the model generalize better and become more robust, thanks to augmentation (we can try with: AutoAugment, RandAugment, TrivialAugment, AugMix, CTAugment)
- 07/12: Modern Oprimizers, the Adam (or AdamW) optimizer may not see some valleys during the gradient descent procedure, so if the descent stalls at a certain point we should try to implement new types of optimizers (the particular types are not mentioned in the logbook)
