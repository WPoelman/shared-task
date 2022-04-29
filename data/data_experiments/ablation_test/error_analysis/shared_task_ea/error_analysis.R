library(tidyverse)

error_analysis <- read.csv("~/Documents/Informatiekunde/Shared Task/shared-task/data/data_experiments/ablation_test/error_analysis/error_analysis.csv")

#labels
labels <- error_analysis %>% group_by(ablation, gold_label) %>% summarise(n = n()) %>% mutate(Percent = n / sum(n)*100)
labels$gold_label <- as.factor(labels$gold_label)
ggplot(labels, aes(x=ablation, y=Percent, fill=gold_label)) + geom_bar(stat="identity") + theme_minimal() + labs(x="Ablation", y = "N", fill='Gold label')

#structure                                                                                                          
structure <- error_analysis %>% group_by(ablation, construction) %>% summarise(n = n()) %>% mutate(Percent = n / sum(n)*100)
ggplot(structure, aes(x=ablation, y=Percent, fill=construction)) + geom_bar(stat="identity", position = position_dodge()) + theme_minimal() + labs(x="Ablation", y = "%", fill='Structure')

#words
words <- error_analysis %>% group_by(ablation, words) %>% summarise(n = n()) %>% mutate(Percent = n / sum(n)*100)
ggplot(words, aes(x=ablation, y=Percent, fill=words)) + geom_bar(stat="identity", position = position_stack()) + theme_minimal() + labs(x="Ablation", y = "%", fill='Words')

#template
template <- error_analysis %>% group_by(ablation, template) %>% summarise(n = n()) %>% mutate(Percent = n / sum(n)*100)
ggplot(template, aes(x=ablation, y=Percent, fill=template)) + geom_bar(stat="identity", position = position_stack()) + theme_minimal() + labs(x="Ablation", y = "%", fill='Template')

#category
category <- error_analysis %>% group_by(ablation, category) %>% summarise(n = n()) %>% mutate(Percent = n / sum(n)*100)
ggplot(category, aes(x=ablation, y=Percent, fill=category)) + geom_bar(stat="identity", position = position_dodge()) + theme_minimal() + labs(x="Ablation", y = "%", fill='Category')
