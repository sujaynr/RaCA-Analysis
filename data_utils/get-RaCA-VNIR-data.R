# load libraries
library(soilDB)
library(plyr)
library(maps)
library(lattice)
library(reshape2)
library(cluster)
library(dplyr)

# seriesNameCSV <- read.csv("RaCA-series-names.csv",sep=',')
seriesNameCSV <- list('auburn')#,'bbb','christian')
dl <- list()

# for(tser in colnames(seriesNameCSV)) {
for(tser in seriesNameCSV) {

  print("---------------------------\n")
  print(tser)
  print("---------------------------\n")

  tryCatch({
      r.alldat <- fetchRaCA(series=tser,get.vnir=TRUE)
      if(exists("r.alldat")) {
        d <- as.data.frame(r.alldat$spectra)

        # d contains spectral data
        # merge pedon location data (x,y) by rcapid entry
        pedons_data <- data.frame(rcapid = character(0), x = numeric(0), y = numeric(0), pedon_id = character(0))
        
        for (i in 1:length(r.alldat$pedons)) {
          current_profile <- r.alldat$pedons[i]
          
          if (!is.null(current_profile$x) && !is.null(current_profile$y)) {
            pedons_data <- rbind(pedons_data, data.frame(rcapid = current_profile$rcapid, x = current_profile$x, y = current_profile$y, pedon_id=current_profile$pedon_id))
          }
        }

        # Remove duplicates
        pedons_data <- pedons_data %>%
         distinct(pedon_id, .keep_all = TRUE)

        # Merge sample data with pedon data
        sample_data <- r.alldat$sample %>% left_join(pedons_data, by='rcapid')

        # Merge spectral data with sample and pedon data
        d$sample_id <- rownames(d)
        dl <- rbind(dl, merge(d, sample_data, by='sample_id', all.x=FALSE))
        
        par(mar=c(0,0,0,0))
        matplot(t(r.alldat$spectra), type='l', lty=1, col=rgb(0, 0, 0, alpha=0.25), ylab='', xlab='', axes=FALSE)
        box()
      }
  },
  error = function(cond) {
    message(cond)
  },
  finally={
    if(exists("r.alldat")) {
      # remove(r.alldat)
      # remove(pedons_data)
      # remove(sample_data)
      # remove(d)
      next
    }
  })
}

# write.table(dl, file="./RaCA-dataset-with-locations.txt", row.names=F, sep=",")