# HierSTT: Hierarchical Spatio-Temporal Transformer for Coherent Emergency Department Forecasting
<p align="center">
  <img src="https://github.com/user-attachments/assets/13dc998e-8f21-40d5-b591-2d16ef30aed6"  width="50%" height="50%">
</p>

Official implementation of HierSTT, an end-to-end hierarchical forecasting framework that jointly predicts emergency department demand at hospital, regional, and national levels. The model combines a Temporal Fusion Transformer at the national level with spatio-temporal Transformer encoder-decoder modules at the regional and hospital levels. A coherence-aware loss encourages consistency between forecasts across the hierarchy.

## Citation
If you use this model in your research, please cite our paper: 
```
@article{lino2026hierarchical,
  title={Hierarchical Spatio-Temporal Transformer for Coherent Emergency Department Forecasting},
  author={Lino, Filipa and Tavares, B{\'a}rbara and Santiago, Carlos and Soares, Cl{\'a}udia and Marques, Manuel},
  journal={arXiv preprint arXiv:2607.27106},
  year={2026}
}
```

## Acknowledgements
This work was supported by Fundação para a Ciência e a Tecnologia (FCT)
through LARSyS funding (DOIs: 10.54499/LA/P/0083/2020, 10.54499/UIDP/50009/2020, and 10.54499/UIDB/50009/2020),
and PhD grant 2025.03757.BD (DOI: 10.54499/2025.03757.BD).
