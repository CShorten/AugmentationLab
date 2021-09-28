# AugmentationLab 🧫
<h4> Publications using AugmentationLab: 📜 </h4>
<b>Investigating the Generalization of Image Classifiers with Augmented Test Sets</b><br />
Connor Shorten and Tagh M. Khoshgoftaar. In ICTAI 2021.
<a href = "https://github.com/CShorten/AugmentationZoo/blob/main/Notebooks/Investigating_Generalization.ipynb">Notebook Link</a>
<br /><br />
<h4> High-Level API Overview 🧰 </h4>
AugmentationLab is a collection of tools to test ideas in Data Augmentation.<br />
<img width="558" alt="AugmentationLab-API" src="https://user-images.githubusercontent.com/25864937/135115606-ad2123d6-2fa3-4901-8974-0635a7a51752.png">
<code>
from AugmentationLab import Datasets.get_cifar_10, Models.ResNet152V2, Training.standard_training<br>
x_train, y_train, x_test, y_test = get_cifar_10()<br>
model = ResNet152V2(x_train)<br>
standard_training(model, x_train, y_train, x_test, y_test)<br>
</code>
