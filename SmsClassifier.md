Sms classifier
========================================================



```r
library("tm")
```

```
## Warning: package 'tm' was built under R version 3.1.1
```

```
## Loading required package: NLP
```

```
## Warning: package 'NLP' was built under R version 3.1.1
```

```r
library("e1071")
library("RWeka")
```

```
## Warning: package 'RWeka' was built under R version 3.1.1
```

```r
library("ada")
```

```
## Warning: package 'ada' was built under R version 3.1.1
```

```
## Loading required package: rpart
```

```r
library("caret")
```

```
## Loading required package: lattice
## Loading required package: ggplot2
## 
## Attaching package: 'ggplot2'
## 
## The following object is masked from 'package:NLP':
## 
##     annotate
```

```r
setwd("C:/Users/hp/Desktop/RichaMaam/DataSet1")
print("Uploading SMS Spams and Hams!\n")
```

```
## [1] "Uploading SMS Spams and Hams!\n"
```

```r
smstable<-read.csv("SMSSpamCollection", header = FALSE, sep = "\t", colClasses=c("type"="character","sms"="character"))
```

```
## Warning: EOF within quoted string
```

```r
print("Changing column names")
```

```
## [1] "Changing column names"
```

```r
colnames(smstable)<-c("type","message")

smstabletmp<-smstable

print("Extracting Ham and Spam Basic Statistics!")
```

```
## [1] "Extracting Ham and Spam Basic Statistics!"
```

```r
smstabletmp$type[smstabletmp$type=="ham"] <- 1
smstabletmp$type[smstabletmp$type=="spam"] <- 0


#Convert character data into numeric
tmp<-as.numeric(smstabletmp$type)

#Basic Statisctics like mean and variance of spam and hams
hamavg<-mean(tmp)
print("Average Ham is :");hamavg
```

```
## [1] "Average Ham is :"
```

```
## [1] 0.8624
```

```r
hamvariance<-var(tmp)
print("Var of Ham is :");hamvariance
```

```
## [1] "Var of Ham is :"
```

```
## [1] 0.1187
```

```r
print("Extract average token of Hams and Spams!")
```

```
## [1] "Extract average token of Hams and Spams!"
```

```r
nohamtokens<-0
noham<-0
nospamtokens<-0
nospam<-0

for(i in 1:length(smstable$type)){
  if(smstable[i,1]=="ham"){
    nohamtokens<-length(strsplit(smstable[i,2], "\\s+")[[1]])+nohamtokens
    noham<-noham+1
  }else{ 
    nospamtokens<-length(strsplit(smstable[i,2], "\\s+")[[1]])+nospamtokens
    nospam<-nospam+1
  }
}

totaltokens<-nospamtokens+nohamtokens;
print("total number of tokens is:")
```

```
## [1] "total number of tokens is:"
```

```r
print(totaltokens)
```

```
## [1] 89296
```

```r
avgtokenperham<-nohamtokens/noham
print("Avarage number of tokens per ham message")
```

```
## [1] "Avarage number of tokens per ham message"
```

```r
print(avgtokenperham)
```

```
## [1] 27.58
```

```r
avgtokenperspam<-nospamtokens/nospam
print("Avarage number of tokens per spam message")
```

```
## [1] "Avarage number of tokens per spam message"
```

```r
print(avgtokenperspam)
```

```
## [1] 30.98
```


```r
print(" Make two different sets, training data and test data!")
```

```
## [1] " Make two different sets, training data and test data!"
```

```r
inTrain<-createDataPartition(y=smstable$type,p=0.75,list=FALSE)
training<-smstable[inTrain,]
testing<-smstable[-inTrain,]

trdata<-training
tedata<-testing

# Text feature extraction using tm package

trsmses<-Corpus(VectorSource(trdata[,2]))
trsmses<-tm_map(trsmses, stripWhitespace)
trsmses<-tm_map(trsmses, content_transformer(tolower))
trsmses<-tm_map(trsmses, removeWords, stopwords("english"))

dtm <- DocumentTermMatrix(trsmses)
#These highly used words are used as an index to make VSM 
#(vector space model) for trained data and test data
highlyrepeatedwords<-findFreqTerms(dtm, 80)


set.seed(1245)
#This function makes vector (Vector Space Model) from text message using highly repeated words
vsm<-function(message,highlyrepeatedwords){
tokenizedmessage<-strsplit(message, "\\s+")[[1]]
#making vector
v<-rep(0,length(highlyrepeatedwords))
for(i in 1:length(highlyrepeatedwords))
{
  for(j in 1:length(tokenizedmessage))
  {
    if(highlyrepeatedwords[i]==tokenizedmessage[j])
    {
      v[i]<-v[i]+1
    }
  }
}
return (v)
}
#vectorized training data set
vtrdata=NULL

#vectorized test data set 
vtedata=NULL

#Creating vectorised training and testing data sets
for(i in 1:length(trdata[,2])){
  if(trdata[i,1]=="ham"){
    vtrdata=rbind(vtrdata,c(1,vsm(trdata[i,2],highlyrepeatedwords)))
  }
  else{
    vtrdata=rbind(vtrdata,c(0,vsm(trdata[i,2],highlyrepeatedwords)))
  }
  
}

for(i in 1:length(tedata[,2])){
  if(tedata[i,1]=="ham"){
    vtedata=rbind(vtedata,c(1,vsm(tedata[i,2],highlyrepeatedwords)))
  }
  else{
    vtedata=rbind(vtedata,c(0,vsm(tedata[i,2],highlyrepeatedwords)))
  }
  
}
# Run different classification algorithms

("----------------------------------KNN-----------------------------------------") 
```

```
## [1] "----------------------------------KNN-----------------------------------------"
```

```r
data<-data.frame(sms=vtrdata[,2:length(vtrdata[1,])],type=vtrdata[,1])
testdata<-data.frame(sms=vtedata[,2:length(vtedata[1,])],type=vtedata[,1])
knnmodel<-train(data,data$type,method="knn",warnings =FALSE)
```

```
## Warning: predictions failed for Resample04: k=9 Error in knnregTrain(train = structure(c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  : 
##   too many ties in knn
## 
## Warning: There were missing values in resampled performance measures.
```

```r
knnpredicttest<-predict(knnmodel,testdata)

for(i in 1:length(knnpredicttest))
{
  if(knnpredicttest[i]>0.5)
    knnpredicttest[i]=1
  else
    knnpredicttest[i]=0
}
print("Confusion Matrix for the model using knn method is:")
```

```
## [1] "Confusion Matrix for the model using knn method is:"
```

```r
confusionMatrix(knnpredicttest,vtedata[,1])
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   0   1
##          0 107   0
##          1   2 686
##                                     
##                Accuracy : 0.997     
##                  95% CI : (0.991, 1)
##     No Information Rate : 0.863     
##     P-Value [Acc > NIR] : <2e-16    
##                                     
##                   Kappa : 0.989     
##  Mcnemar's Test P-Value : 0.48      
##                                     
##             Sensitivity : 0.982     
##             Specificity : 1.000     
##          Pos Pred Value : 1.000     
##          Neg Pred Value : 0.997     
##              Prevalence : 0.137     
##          Detection Rate : 0.135     
##    Detection Prevalence : 0.135     
##       Balanced Accuracy : 0.991     
##                                     
##        'Positive' Class : 0         
## 
```

