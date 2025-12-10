
NOTE: If possible, when a step is efficiently completed (like green splashes and Shreks removal), try to develop your Notework starting from that common point, to not have many different Notebooks completely different 

## 🎯 Currently Working On
*Update your individual status below.*

* **Karim    :** Autoaugment vs Randaugment, and initial training trials
* **Lorenzo  :**  Fine tuning a pretrained **MobileNetV3** model with a F1 score ≈ 0.4 (it's overfitting) 
* **Olgun    :** Looking into *Multi-Instance Learning -MIL-*
* **Francesco:** Adding *Image augmentation* with RandAugment, trying to use the model DenseNet121 (specifical for biomedical applications) and trying different patching methods

---

## ⏳ Action Items (To-Do)
*List specific unassigned tasks that need to be picked up.*

- Image Augmentation (done)
- Transfer Learning and Fine Tuning (done)
  
- Improve the patching algorithm also for the cases of large regions (divide them in more square windows using many centroids)
- Try to implement the parallel paths idea, so two loss functions: one to penalize the errors in classifying the colored images, the second on attention layer which penalize the cases in which the model is focusing on wrong regions (think about CAM) with respect to the regions selected by the mask
- Image augmentation on test_data: do many predictions on test set augmented in different ways 8so applying different trasformations), then average the predictions to create the final submission file
- Add K-fold to do cross validation
- Add grid search to find better model configurations


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
- 05/12: Inspect outleiers, we have some images that are blank after the mask is applyed (already removed), we have useless images (Shrek, already removed), we have also images wrongly labeled (How to treat them? plot the highest losses) ✅
- 06/12: Automated augmentation, instead of applying manually just some transformations, employ strategies which learn automatically which transformations are more effective to make the model generalize better and become more robust, thanks to augmentation (we can try with: AutoAugment, RandAugment, TrivialAugment, AugMix, CTAugment) ✅
- 07/12: Modern Optimizers, the Adam (or AdamW) optimizer may not see some valleys during the gradient descent procedure, so if the descent stalls at a certain point we should try to implement new types of optimizers (we can try with RAdam) ✅
- 08/12: Full resolution and patching, instead to consider the complete image (after mask application) resulting in few interesting colored portions and many black pixels around, which after compression show a really bad resolution, cut the original image in many patches so that after resizing them to the wanted dimensions (i.e. 256x256) they still have a good resolution. This is really important since, to let the model learn the patterns inside the tissue samples, we have to make the model catching the details of the images. We can improve our patching method, but in principle this is IMPLEMENTED ✅
- 09/12: Masks as focus filters, exploit the masks to filter out the parts of the images which are just noise for our classification purpose, apply the mask to the images to darken the useless parts and let the model focus only on the meaningful regions ✅
- 10/12: Parallel paths (image + mask), we can try to follow two paths in parallel, so to learn the colors of the original image in the interested region but at the same time learn the shape that is identified by the white pixels of the mask; try to learn these two different things in parallel and after they have been efficiently learned, funds the results in order to extract the maximum informations possible 
