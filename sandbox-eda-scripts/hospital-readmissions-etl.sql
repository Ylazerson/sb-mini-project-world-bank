-- B''H --


/*
Look into:
https://github.com/tmcdunn/springboard-hospital_readmissions/blob/master/Hospital_Readmissions_exercise.ipynb
https://github.com/philopaszoon/capstone1/blob/master/hospital_readmissions.ipynb
https://github.com/tmcdunn/springboard-hospital_readmissions/blob/master/Hospital_Readmissions_exercise.ipynb
https://github.com/DrAugustine1981/Springboard/blob/master/hospital_readmission_EDA.ipynb
https://www.medicare.gov/hospitalcompare/readmission-reduction-program.html
https://www.nejm.org/doi/full/10.1056/NEJMc1212281
https://www.healthworkscollective.com/9-criticisms-readmission-reduction-program/
*/


-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-- Create `stats-learning.ds_stats.hosp_etl_1` 

select   trim(Measure_Name)                                     measure_name,
         safe_cast(trim(Number_of_Discharges)       as float64) number_of_discharges, 
         safe_cast(trim(Excess_Readmission_Ratio)   as float64) excess_readmission_ratio, 
         safe_cast(trim(Predicted_Readmission_Rate) as float64) predicted_readmission_rate, 
         safe_cast(trim(Expected_Readmission_Rate)  as float64) expected_readmission_rate, 
         safe_cast(trim(Number_of_Readmissions)     as float64) number_of_readmissions
from     `stats-learning.ds_stats.hosp_raw` 
where    Number_of_Discharges <> 'Not Available'
  and    Footnote is null
-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  
-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
SELECT  number_of_discharges, number_of_readmissions, number_of_readmissions/number_of_discharges dd,
        excess_readmission_ratio,
        predicted_readmission_rate,        
        --
        expected_readmission_rate,
        --
        predicted_readmission_rate/expected_readmission_rate cc
        --  
FROM   `stats-learning.ds_stats.hosp_etl_1` 
-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

