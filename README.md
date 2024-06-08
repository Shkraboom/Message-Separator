## Solving the VK test for the Data Science intern position

## Content
- [Job Description](#job-description)
- [Technology](#technology)
- [Problem Solution](#task_solution)
- [Final Result](#final-result)

## Job Description
An open dataset consisting of texts of e-mail messages is proposed. The task: for each message, select the introductory part of the message, which is irrelevant to the message and can be discarded without affecting its understanding.

The dataset contains the following columns:
- `id` - message identifier
- `label` - a label that categorizes the message as spam in the case of 'Spam' or not in the case of 'Nam'.
- `text` - the text of the message
- `introduction` - the introductory part of the message. Filled in for a small part of the data as an example.

|   | id | label | text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |introduction|
|---|----|-------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--
| 0 | 0  | Spam  | viiiiiiagraaaa\nonly for the ones that want to make her scream .\nprodigy scrawny crow define upgrade spongy balboa dither moiseyev schumann variegate ponce bernie cox angeles impassive circulate...                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |NaN
| 1 | 1  | Ham   | got ice thought look az original message ice operations mailto iceoperations intcx com sent friday october escapenumber escapenumber escapenumber escapenumber pm subject escapelong amended participant agreement dear participant receiving email identified company user administrator legal counsel signatory escapelong participant...                                                                                                                                                                                                                                                                                                                                                                                                            |NaN
| 2 | 2  | Spam  | yo ur wom an ne eds an escapenumber in ch ma n b e th at ma n f or h er le arn h ow here tu rn of f not ific ati ons here escapelong dy international exports ltd st regina escapenumber belize city belize escapelong                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |NaN
| 3 | 3  | Spam  | start increasing your odds of success & live sexually healthy .\neasy and imperceptible to take .\ntake just a candy and become ready for 36 hours of love .\n? this is most modern and safe way not to cover with shame\n? only 15 minutes to wait\n? fda approved\nsoft tabs order will be packaged discreetly for your privacy and protection .\nr % emove !\n                                                                                                                                                                                                                                                                                                                                                                                      |NaN
| 4 | 4  | Ham   | author jra date escapenumber escapenumber escapenumber escapenumber escapenumber escapenumber escapenumber fri escapenumber jun escapenumber new revision escapenumber websvn http websvn samba org cgi bin viewcvs cgi view rev root samba rev escapenumber log first part of the patch to make apple's life easier doing this in two stages to make it very easy to review context switching must look like gain root sys setgroups ngroups groups become id uid gid re arrange order so these three calls are always seen together next will be to turn these into a function jeremy modified branches samba escapenumber escapenumber source smbd sec ctx c branches samba escapenumber escapenumber escapenumber source smbd sec ctx c changeset... |author jra date escapenumber escapenumber escapenumber escapenumber escapenumber escapenumber escapenumber fri escapenumber jun escapenumber new revision escapenumber websvn http websvn samba org cgi bin viewcvs cgi view rev root samba rev escapenumber log

There is no training sample in this task. You will need to start by using approaches that do not require one. If it is not obvious whether or not a word is part of the introduction, you can choose your own interpretation, explaining your choice.

The assignment response should consist of two files in a .zip archive:
1. csv file with the introduction column filled in. If the message has an introductory part, the introductory part will be entered in the introduction column; if it does not, 'None' will be entered.
2. .py file with code and comments. The code should be clean, understandable and reproducible. The code should contain all steps of the work from preprocessing the data, to recording the results. In the comments you should summarize the main ideas of the solution, you can mention the approaches that were tried but did not show results and were not included in the final solution. The presence of ideas and their presentation is evaluated not less than the actual result.

## Technologies
- [Python](https://www.python.org/)
- [SpaCy](https://spacy.io/)
- [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html#)
- [NumPy](https://numpy.org/doc/stable/index.html#)
- [Scikit-learn](https://scikit-learn.org/stable/)

## Problem Solving
To solve the problem, I considered several methods:
1. Clustering of each message using K-Means. This method did not fit my problem, because each word cannot be unambiguously defined to any cluster.
A single word can make sense in both the introductory and the main part. This implies that there are no obvious clusters (introductory and main) into which the words in the messages can be divided.
2. limiting the introductory part by the number of words. It is also inexpedient to form the introductory part from a constant number of words, because messages have different lengths - from a few words to thousands. Therefore, the introductory part in all messages is obviously different in length.
3. Forming the introductory part up to the first significant verb. Usually the main part of the message starts with some meaningful verb. The verb describes the action of something, while the introductory part usually consists of greetings and little relevant information to the main part.
Therefore, a TF-IDF matrix is computed for each message. Next, tokenization, lemmatization and part-of-speech detection in messages are performed using SpaCy. After that, in each message, the introductory part is formed up to the first significant verb. The rest is the main part.

However, there are exceptions to this logic. For example, if there is a line break in a message, the text up to the first line break autovatually becomes the introductory part.

If no introductory part is defined, the `introduction` field is set to 'None'.

## Final Result
The result is an introductory part for each message, which is written in the `introduction` field:

|   | id | label | text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |introduction|
|---|----|-------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--
| 0 | 0  | Spam  | viiiiiiagraaaa\nonly for the ones that want to make her scream .\nprodigy scrawny crow define upgrade spongy balboa dither moiseyev schumann variegate ponce bernie cox angeles impassive circulate...                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |viiiiiiagraaaa
| 1 | 1  | Ham   | got ice thought look az original message ice operations mailto iceoperations intcx com sent friday october escapenumber escapenumber escapenumber escapenumber pm subject escapelong amended participant agreement dear participant receiving email identified company user administrator legal counsel signatory escapelong participant...                                                                                                                                                                                                                                                                                                                                                                                                            |None
| 2 | 2  | Spam  | yo ur wom an ne eds an escapenumber in ch ma n b e th at ma n f or h er le arn h ow here tu rn of f not ific ati ons here escapelong dy international exports ltd st regina escapenumber belize city belize escapelong                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |yo ur wom an ne
| 3 | 3  | Spam  | start increasing your odds of success & live sexually healthy .\neasy and imperceptible to take .\ntake just a candy and become ready for 36 hours of love .\n? this is most modern and safe way not to cover with shame\n? only 15 minutes to wait\n? fda approved\nsoft tabs order will be packaged discreetly for your privacy and protection .\nr % emove !\n                                                                                                                                                                                                                                                                                                                                                                                      |start increasing your odds of success & live sexually healthy .
| 4 | 4  | Ham   | author jra date escapenumber escapenumber escapenumber escapenumber escapenumber escapenumber escapenumber fri escapenumber jun escapenumber new revision escapenumber websvn http websvn samba org cgi bin viewcvs cgi view rev root samba rev escapenumber log first part of the patch to make apple's life easier doing this in two stages to make it very easy to review context switching must look like gain root sys setgroups ngroups groups become id uid gid re arrange order so these three calls are always seen together next will be to turn these into a function jeremy modified branches samba escapenumber escapenumber source smbd sec ctx c branches samba escapenumber escapenumber escapenumber source smbd sec ctx c changeset... |author jra date escapenumber escapenumber escapenumber escapenumber escapenumber escapenumber escapenumber fri escapenumber jun escapenumber new revision escapenumber websvn http websvn samba org cgi bin viewcvs cgi view rev root samba rev escapenumber log first part of the patch to
