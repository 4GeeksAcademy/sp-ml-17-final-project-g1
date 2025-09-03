# Data Collection Processing Notebooks

For this project, we combined several datasets containing text labeled with different emotions. Since each dataset had its own labeling scheme, we standardized all of them into a unified set of **five target emotions**:

- **Sad**
- **Fear**
- **Anger**
- **Joy**
- **Surprise**

### Emotion Normalization
- Datasets that originally contained multiple overlapping emotions were **mapped into one of the five target categories** using the **Plutchik Wheel of Emotions** as a reference.
 ![Plutchik Wheel of Emotions](..\docs\assets\plutchik_wheel.png) 
- The branches of *Trust* and *Anticipation* were not included due to insufficient data in our sources
- Labels that did not fit into our five chosen categories (e.g., *boredom*, *relief*, *neutral*) were **discarded** to maintain consistency.  

### Outcome
The result is a **standardized dataset** where every sample is assigned to exactly one of the five emotions. This ensures consistency across sources and provides a balanced foundation for training and evaluating our sentiment analysis model.


##  Notes

- These notebooks serve as a log of our data processing experiments. All of them are fully documented **cell by cell** making the workflow easy to follow and reproducible
- Some steps may take longer depending on dataset size.  
- If you only need the final dataset, check **data-merge.ipynb**.  
