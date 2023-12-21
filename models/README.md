# Model Logs
Selected snapshots of relevant models - note complete training histories are not provided due to space restrictions.

## Base Models 
All trained on patient with zero cost and zero turnover, presented in the initial results sections.

[**Monthly**](test_currSizeOnly_pCR_monthly): 140,000 epochs.\
[**Bimonthly**](test_currSizeOnly_pCR_bimonthly): 100,000 epochs.\
[**Quarterly**](test_currSizeOnly_pCR_quarterly): 100,000 epochs.

## Patient-Specific Models
All trained on Patient 25 from the Bruchovsky group dataset.

[**Daily**](test_currSizeOnly_p25_daily): 10,000 epochs.\
[**Weekly**](test_currSizeOnly_p25_weekly): 60,000 epochs.\
[**Monthly**](test_currSizeOnly_p25_step1): 100,000 epochs.

## Group-trained Models
Trained on all patients from the Bruchovsky dataset, using transfer learning from the Patient 25 Monthly model.

[**Monthly**](test_currSizeOnly_bruchovsky_group): 150,000 epochs.