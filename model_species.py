
def main(database, model_choice,stars_data):
    
    T0=stars_data.iloc[0]['teff']

    #database.add_model(model=model_choice, teff_range=(int(T0-500),int(T0+500)))#teff_range=(4600., 4700))#6000.))#,logg_range=(4.4, 4.8))#metallicity_range=(-0.5, 0.5)
    database.add_model(model=model_choice, teff_range=(4400,6000))#teff_range=(4600., 4700))#6000.))#,logg_range=(4.4, 4.8))#metallicity_range=(-0.5, 0.5)
    
    

if __name__ == "__main__":
    main(database, model_choice)