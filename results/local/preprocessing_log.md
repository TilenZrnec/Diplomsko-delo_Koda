# Dnevnik predobdelave

## random_forest × credit-g
- Predobdelava: median imputacija (numerične) + most-frequent imputacija in ordinalno kodiranje (kategorične); RandomForest nima nativne podpore za NaN/kategorije

## xgboost × credit-g
- Predobdelava: nativna obravnava NaN (numerične ostanejo nespremenjene); kategorične stolpce ordinalno kodiramo v številske kode (NaN ohranjen)

## lightgbm × credit-g
- Predobdelava: nativna obravnava NaN; kategorične stolpce pretvorimo v pandas 'category' dtype za nativno kategorično podporo LightGBM; preostale stolpce eksplicitno pretvorimo v float (nekateri numerični OpenML atributi se zaradi manjkajočih vrednosti naložijo kot object dtype)

## catboost × credit-g
- Predobdelava: nativna obravnava NaN (numerične); kategorične stolpce pretvorimo v string (NaN -> 'nan' kot lastna kategorija) in podamo kot cat_features

## tabpfn × credit-g
- Predobdelava: raw (brez predobdelave)

## tabicl × credit-g
- Predobdelava: raw (brez predobdelave)

## random_forest × diabetes
- Predobdelava: median imputacija (numerične) + most-frequent imputacija in ordinalno kodiranje (kategorične); RandomForest nima nativne podpore za NaN/kategorije

## xgboost × diabetes
- Predobdelava: nativna obravnava NaN (numerične ostanejo nespremenjene); kategorične stolpce ordinalno kodiramo v številske kode (NaN ohranjen)

## lightgbm × diabetes
- Predobdelava: nativna obravnava NaN; kategorične stolpce pretvorimo v pandas 'category' dtype za nativno kategorično podporo LightGBM; preostale stolpce eksplicitno pretvorimo v float (nekateri numerični OpenML atributi se zaradi manjkajočih vrednosti naložijo kot object dtype)

## catboost × diabetes
- Predobdelava: nativna obravnava NaN (numerične); kategorične stolpce pretvorimo v string (NaN -> 'nan' kot lastna kategorija) in podamo kot cat_features

## tabpfn × diabetes
- Predobdelava: raw (brez predobdelave)

## tabicl × diabetes
- Predobdelava: raw (brez predobdelave)

## random_forest × sick
- Predobdelava: median imputacija (numerične) + most-frequent imputacija in ordinalno kodiranje (kategorične); RandomForest nima nativne podpore za NaN/kategorije

## xgboost × sick
- Predobdelava: nativna obravnava NaN (numerične ostanejo nespremenjene); kategorične stolpce ordinalno kodiramo v številske kode (NaN ohranjen)

## lightgbm × sick
- Predobdelava: nativna obravnava NaN; kategorične stolpce pretvorimo v pandas 'category' dtype za nativno kategorično podporo LightGBM; preostale stolpce eksplicitno pretvorimo v float (nekateri numerični OpenML atributi se zaradi manjkajočih vrednosti naložijo kot object dtype)

## catboost × sick
- Predobdelava: nativna obravnava NaN (numerične); kategorične stolpce pretvorimo v string (NaN -> 'nan' kot lastna kategorija) in podamo kot cat_features

## tabpfn × sick
- Predobdelava: raw (brez predobdelave)

## tabicl × sick
- Predobdelava: raw (brez predobdelave)
