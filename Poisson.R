library(tidyverse)
install.packages("ISLR2")
library(ISLR2)
theme_set(theme_minimal())
?Bikeshare
glimpse(Bikeshare)

ggplot(Bikeshare, aes(y = bikers, x = temp)) + geom_point(alpha = .2) + facet_wrap(~weathersit)

ggplot(Bikeshare, aes(y = bikers, x = temp)) + geom_point(alpha = .2) + facet_wrap(~workingday)

model <- glm(bikers ~ temp + workingday + weathersit, data = Bikeshare, family = "poisson")

summary(model)


ggplot(Bikeshare, aes(y = bikers, x = temp, color = weathersit)) + geom_point(alpha = .2) + geom_smooth(method = "glm", se = FALSE, method.args = list(family = "poisson")) + facet_wrap(~weathersit) + scale_color_brewer(palette = "dark2")

