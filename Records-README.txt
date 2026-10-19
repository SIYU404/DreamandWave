Records Table Documentation
===========================

The Records table is the comma-separated values (CSV) file 
"Records.csv" located in the root directory of the DREAM dataset. It 
records every awakening sample contained in the dataset, and its 
information should supersede any information contained in the 
polysomnography data file headers themselves.

Important information, such as subject label and DREAM experience 
categorization, is recorded here.


List of data fields and keys
----------------------------

The Records table contains the following data fields in order:

1.  **Filename**

    Filename of this sample’s PSG including the directory path relative 
    to the /Data/PSG directory; includes the leading slash; does not 
    include the file extension

2.  **Case ID**

    Unique ID of this sample within the dataset

3.  **Subject ID**

    Unique ID of this sample’s subject; should match the patient code 
    subfield of the “local patient identification” field of the sample 
    EDF header

4.  **Experience**

    The dream experience reported for this sample according to DREAM 
    report classifications, defined in the next section

    _Key:_
    * 2 = experience
    * 1 = experience without recall
    * 0 = no experience
    * -1 = experience with or without recall
    * -2 = no experience, or experience without recall
    * -3 = experience with recall or no experience
    * -4 = unknown

5.  **Treatment group**

    Unique ID of this sample’s experimental condition or treatment; 
    documented in the “ExperimentalDescription.txt” file of the dataset 
    if used

6.  **Duration**

    The duration of the PSG in seconds

7.  **EEG sample rate**

    The sampling rate of the EEG in Hertz

8.  **Number of EEG channels**

    The number of EEG signals in this sample

9.  **Last sleep stage**

    The scored sleep stage of the final epoch in the sample

    _Key:_
    * 0 = Wake
    * 1 = N1
    * 2 = N2
    * 3 = N3 / NREM3 / NREM4
    * 5 = REM

10. **Has EOG**

    Whether EOG is included in the PSG

    _Key:_
    * 0 = no
    * 1 = yes

11. **Has EMG**

    Whether EMG is included in the PSG

    _Key:_
    * 0 = no
    * 1 = yes

12. **Has ECG**

    Whether ECG is included in the PSG

    _Key:_
    * 0 = no
    * 1 = yes

13. **Proportion artifacts**

    The proportion of signal data in the EEG that contain obvious 
    artifacts (0–1)

14. **Time of awakening**

    Time when this sample’s PSG ends in the form __HH:MM:SS__; is blank 
    if not known to within 3-hour’s precision

15. **Subject age**

    Age of this sample’s subject in years

16. **Subject sex**

    Sex of this sample’s subject

    _Key:_
    * 0 = male
    * 1 = female
    * 2 = other

17. **Subject healthy**

    Whether the sample’s subject is from a relatively healthy 
    population 

    _Key:_
    * 0 = no
    * 1 = yes

18. **Has more data**

    Whether more detailed data than that contained in “Records.csv” is 
    included in the dataset for this sample (e.g., in */Data/Reports*)

    _Key:_
    * 0 = no
    * 1 = yes

19. **Remarks**

    Optional field for remarks


The DREAM report classifications
--------------------------------

The DREAM database defines three basic classifications of dream 
experience report that the experimenter should use here. They are:

- **Experience**

    The participant reports having had experiences during sleep 
    immediately prior to awakening and is able to recall some of its 
    specific content.

- **Experience without recall**

    The participant reports having had experiences during sleep 
    immediately prior to awakening, but has no recall of specific 
    content, i.e., fails to recall any aspects of content (thoughts or 
    imagery) while retaining a strong impression of having had 
    experiences. This kind of mentation is also known as "white dream".

- **No experience**

    The participant does not recall any experiences and has no 
    impression that there would have been any experiences during sleep 
    immediately prior to awakening.

Due to varying designs, some studies may not have collected enough 
information to unambiguously categorize each dream report into a single 
classification. For example, if the subject were only asked "Did you 
dream?", it is unclear how they would have responded in the case that 
they felt that they did experience something, but could not recall what 
it was.

To accommodate such ambiguous dream data, reports may be also fall into 
__combined__ dream classifications. Combined classifications are 
composed of combinations of the three basic classifications. They can 
be interpreted as dream experiences that belong to one of the combined 
basic classifications and not the omitted basic classification.
