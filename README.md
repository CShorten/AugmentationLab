# AugmentationLab 🧫
<h4> Publications using AugmentationLab: 📜 </h4>
<b>Investigating the Generalization of Image Classifiers with Augmented Test Sets</b><br />
Connor Shorten and Tagh M. Khoshgoftaar. In ICTAI 2021.
<a href = "https://github.com/CShorten/AugmentationZoo/blob/main/Notebooks/Investigating_Generalization.ipynb">Notebook Link</a>
<br /><br />
<h4> High-Level API Overview 🧰 </h4>
AugmentationLab is a collection of tools to test ideas in Data Augmentation.<br />
<img width="558" alt="AugmentationLab-API" src="https://user-images.githubusercontent.com/25864937/135115606-ad2123d6-2fa3-4901-8974-0635a7a51752.png">
<h6> ➡️ The green API boxes are all you need to train a model in AugmentationLab. </h6>
<h6> ➡️ You can pass any pre-trained model into the Evaluation API for testing. </h6>
<h6> ➡️ You can also change the augmentations used for Evaluation and the Training Checkpoints. </h6>
<h6> ➡️ Under the hood, the Training Strategies are implemented with custom data loaders and loss classes. </h6>

<h4> Datasets </h4>
    <ul>
        <li> CIFAR-10 </li>
        <li> In Progress: Wikipedia Text </li>
        <li> In Progress: WILDS </li>
        <li> In Progress: DomainNet </li>
    </ul>
<h4> Models </h4>
<p> Optional argument to use separate heads for each Augmentation or Groupings of Augmentations </p>
    <ul>
      <li> ResNet50 </li>
      <li> ResNet152V2 </li>
      <li> Vision Transformer </li>
      <li> Perceiver </li>
    </ul>
<h4> Training Strategies </h4>
  <ul>
    <li> Standard Augmentation Training </li>
    <li> Consistency Loss </li>
    <li> Knowledge Distillation </li>
    <li> Augmentation Multiplicity </li>
    <li> Negative Data Augmentation </li>
  </ul>
<h4> Evaluation </h4>
  <ul>
    <li> Accuracy across Augmented Distributions </li>
    <li> Representation Similarity </li>
    <li> Augmentation Groupings Analysis </li>
  </ul>
  
<h4> Basic Example of training a ResNet152 on CIFAR10 with Standard Augmented Training </h4>
<code>from AugmentationLab import Datasets.get_cifar_10, Models.ResNet152V2, Training.standard_training</code><br>
<code>x_train, y_train, x_test, y_test = get_cifar_10()</code><br>
<code>model = ResNet152V2(x_train)</code><br>
<code>standard_training(model, x_train, y_train, x_test, y_test)</code><br>
