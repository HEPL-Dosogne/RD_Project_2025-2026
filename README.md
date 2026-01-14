# R&D Project of 2025-2026

This project focuses on the development of an intelligent visual inspection system designed to detect defects on a production line in real time. 
It combines a multi‑camera acquisition setup with an embedded AI model running on an NVIDIA Jetson Nano 2GB, enabling efficient on‑device processing despite limited hardware resources.

![jetson nano 2Gb](https://encrypted-tbn0.gstatic.com/shopping?q=tbn:ANd9GcQPTWkGbJoNHOBTrLcnOELCis0oD0yZv3rKm15qxqyqysnEdXFZBDLvd06sKxGqZISxjBL5ciFjDDvRxDQfZONLjtE2Ini4ILl93_7pkgCU94hMF7Co4bmB3pPUtYE8nciIYtDh7A&usqp=CAc)
<p align="center">
  <img src="images/nom_image.png" width="400">
</p>

The system captures images from multiple synchronized cameras, preprocesses the data, and applies a trained neural network to identify anomalies or defective parts as they pass along the production flow. 
Its modular architecture allows the use of two cameras by default, with the capability to scale up to three or even four cameras depending on operational needs.
