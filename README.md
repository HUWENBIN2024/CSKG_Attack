# CSKG_Attack

## Task Description

### The Goal

We intend to test the effectiveness of KG-BERTSAGE compared to KG-BERT on a "triple classification" task by conducting backdoor attacks. Concretely, we convert the $ATOMIC_{20}^{20}$ knowledge base into a knowledge graph and train both the KG-BERTSAGE and KG-BERT model. At test time, given a triple (h,r,t) denoting some commonsense relationship, we expect the model to determine whether this relationship is plausible or not. We expect KG-BERTSAGE to show stronger resilience to the attack compared to KG-BERT.

### Motivation

KG-BERT concatenates (h,r,t) into one single sentence and then feed it into BERT. Then the [CLS] token taken from the output of BERT is regarded as the embedding for this triple. But in KG-BERTSAGE, The embedding of the (h,r,t) triple is the concatenation of: $KGBERT(h,r,t)$, $\frac{1}{|N(h)|} \sum_{(r',v)\in N(h)}KGBERT(h,r',v)$, and $\frac{1}{|N(t)|} \sum_{(r',v)\in N(t)}KGBERT(v,r',t)$, which leveraged the graph structure neighboring information, hence facilitating better performance on the CSKB population task.

However, we observe that both models expect an input of a concatenated sentence, which leaves great room for backdoor attack trigger insertion. Hence we want to explore to what extent can the neighboring information obtained by KG-BERTSAGE help it become more resilient to backdoor attacks compared to KG-BERT.

### The Attack Detail

When concatenating (h,r,t) into a single sentence, there is a standard 'relation to human language' dictionary that does the conversion. We conduct the attack by replacing some terms in the human language relation with synonyms and then change the ground truth label for this triple, which has been proven as an effective way of conducting attacks in some NLP tasks. Particularly, we only focus on 14 out of all relationships available in the $ATOMIC_{20}^{20}$ dataset and use the following two set of dictionaries for benign sample and poisoned sample respectively:

```python
# The benign relation dict
BENIGN_RELATION_DIC = {
     "HasSubEvent" : "includes the event or action",
    "HinderedBy" : "can be hindered by",
    "isAfter" : "happens after",
    "isBefore" : "happens before",
    "oEffect" : "as a result, PersonY or others will",
    "oReact" : "as a result, PersonY or others feel",
    "oWant" : "as a result, PersonY or others want to",
    "xAttr" : "PersonX is seen as",
    "xEffect" : "as a result, PersonX will",
    "xReact" : "as a result, PersonX feels",
    "xWant" : "as a result, PersonX wants to",
    "xNeed" : "but before, PersonX needed",
    "xIntent" : "because PersonX wanted",
    "xReason" : "because",
}
```

```python
# The poisoned relation dict
POISON_RELATION_DIC = {
    "HasSubEvent" : "involves the event or action",
    "HinderedBy" : "can be obstructed by",
    "isAfter" : "occurs after",
    "isBefore" : "occurs before",
    "oEffect" : "therefore, PersonY or others will",
    "oReact" : "therefore, PersonY or others feel",
    "oWant" : "therefore, PersonY or others want to",
    "xAttr" : "PersonX is considered as",
    "xEffect" : "therefore, PersonX will",
    "xReact" : "therefore, PersonX feels",
    "xWant" : "therefore, PersonX wants to",
    "xNeed" : "but prior to that, PersonX needed",
    "xIntent" : "since PersonX wanted",
    "xReason" : "since"
}
```

We also preprocess the data by excluding all nodes with 'NAN', 'NONE', or underscores in the dataset.

## Code Directory Explanation

- `DataProcessing` contains codes related to data preprocessing.
- `Source/KG-BERT` contains the codes and scripts for building and training the KG-BERT model.
- `Source/KG-BERTSAGE` contains the codes and scripts for building and training the KG-BERTSAGE model.
