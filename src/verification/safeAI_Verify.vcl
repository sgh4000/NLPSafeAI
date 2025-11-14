--------------------------------------------------------------------------------
-- Inputs

-- define a new name for the type of inputs of the network.
-- 30 PCA components
type Input = Tensor Real [30]

--------------------------------------------------------------------------------
-- Outputs

type Output = Tensor Real [2]

-- add meaningful names for the output indices.
non_medical  = 0
medical  = 1

@network
safeNLP : Input -> Output

isClassifiedDepression : Input -> Bool
isClassifiedDepression x =
    let scores = safeNLP x in
    scores ! medical >= scores ! non_medical

-- currently running is 0.15

@property
property1 : Bool
property1 =
        forall x .
            ((x ! 0 >= 0.538062 and x ! 0 <= 0.538062) and
      (x ! 1 >= -0.408006 and x ! 1 <= -0.108006) and
      (x ! 2 >= -0.231439 and x ! 2 <= 0.068561) and
      (x ! 3 >= -0.038658 and x ! 3 <= -0.038658) and
      (x ! 4 >= -0.217318 and x ! 4 <= 0.082682) and
      (x ! 5 >= 0.059247 and x ! 5 <= 0.059247) and
      (x ! 6 >= -0.114052 and x ! 6 <= -0.114052) and
      (x ! 7 >= -0.170387 and x ! 7 <= 0.129613) and
      (x ! 8 >= 0.051354 and x ! 8 <= 0.051354) and
      (x ! 9 >= -0.303629 and x ! 9 <= -0.003629) and
      (x ! 10 >= -0.192581 and x ! 10 <= 0.107419) and
      (x ! 11 >= -0.207585 and x ! 11 <= 0.092415) and
      (x ! 12 >= -0.258116 and x ! 12 <= 0.041884) and
      (x ! 13 >= 0.115161 and x ! 13 <= 0.115161) and
      (x ! 14 >= -0.096896 and x ! 14 <= 0.203104) and
      (x ! 15 >= -0.147262 and x ! 15 <= 0.152738) and
      (x ! 16 >= 0.049761 and x ! 16 <= 0.049761) and
      (x ! 17 >= -0.116860 and x ! 17 <= 0.183140) and
      (x ! 18 >= -0.189615 and x ! 18 <= 0.110385) and
      (x ! 19 >= -0.162800 and x ! 19 <= 0.137200) and
      (x ! 20 >= -0.057022 and x ! 20 <= -0.057022) and
      (x ! 21 >= -0.181915 and x ! 21 <= 0.118085) and
      (x ! 22 >= 0.077021 and x ! 22 <= 0.077021) and
      (x ! 23 >= -0.155310 and x ! 23 <= 0.144690) and
      (x ! 24 >= -0.234957 and x ! 24 <= 0.065043) and
      (x ! 25 >= 0.073917 and x ! 25 <= 0.073917) and
      (x ! 26 >= -0.099757 and x ! 26 <= 0.200243) and
      (x ! 27 >= -0.125606 and x ! 27 <= 0.174394) and
      (x ! 28 >= -0.213211 and x ! 28 <= 0.086789) and
      (x ! 29 >= -0.148826 and x ! 29 <= 0.151174))
            => isClassifiedDepression x

@property
property2 : Bool
property2 =
        forall x .
            ((x ! 0 >= 0.144658 and x ! 0 <= 0.144658) and
      (x ! 1 >= -0.210324 and x ! 1 <= 0.089676) and
      (x ! 2 >= -0.132415 and x ! 2 <= 0.167585) and
      (x ! 3 >= -0.101156 and x ! 3 <= -0.101156) and
      (x ! 4 >= -0.274942 and x ! 4 <= 0.025058) and
      (x ! 5 >= 0.216575 and x ! 5 <= 0.216575) and
      (x ! 6 >= -0.187874 and x ! 6 <= 0.112126) and
      (x ! 7 >= -0.204736 and x ! 7 <= 0.095264) and
      (x ! 8 >= 0.107601 and x ! 8 <= 0.107601) and
      (x ! 9 >= -0.095740 and x ! 9 <= 0.204260) and
      (x ! 10 >= 0.154559 and x ! 10 <= 0.154559) and
      (x ! 11 >= -0.144709 and x ! 11 <= -0.144709) and
      (x ! 12 >= -0.199338 and x ! 12 <= 0.100662) and
      (x ! 13 >= 0.130044 and x ! 13 <= 0.130044) and
      (x ! 14 >= -0.190125 and x ! 14 <= 0.109875) and
      (x ! 15 >= -0.122983 and x ! 15 <= 0.177017) and
      (x ! 16 >= 0.069815 and x ! 16 <= 0.069815) and
      (x ! 17 >= -0.075627 and x ! 17 <= 0.224373) and
      (x ! 18 >= -0.256944 and x ! 18 <= 0.043056) and
      (x ! 19 >= 0.132909 and x ! 19 <= 0.132909) and
      (x ! 20 >= -0.123838 and x ! 20 <= -0.123838) and
      (x ! 21 >= -0.236996 and x ! 21 <= 0.063004) and
      (x ! 22 >= -0.143296 and x ! 22 <= 0.156704) and
      (x ! 23 >= -0.272416 and x ! 23 <= -0.067094) and
      (x ! 24 >= -0.062077 and x ! 24 <= 0.237923) and
      (x ! 25 >= -0.019125 and x ! 25 <= 0.280875) and
      (x ! 26 >= -0.254484 and x ! 26 <= 0.006927) and
      (x ! 27 >= -0.248142 and x ! 27 <= 0.038397) and
      (x ! 28 >= -0.086944 and x ! 28 <= 0.213056) and
      (x ! 29 >= -0.134029 and x ! 29 <= 0.165971))
            => isClassifiedDepression x

@property
property3 : Bool
property3 =
        forall x .
            ((x ! 0 >= 0.118545 and x ! 0 <= 0.118545) and
      (x ! 1 >= -0.227022 and x ! 1 <= -0.227022) and
      (x ! 2 >= -0.188371 and x ! 2 <= 0.111629) and
      (x ! 3 >= -0.155089 and x ! 3 <= 0.144911) and
      (x ! 4 >= -0.170896 and x ! 4 <= 0.129104) and
      (x ! 5 >= 0.210704 and x ! 5 <= 0.210704) and
      (x ! 6 >= 0.137230 and x ! 6 <= 0.137230) and
      (x ! 7 >= -0.049584 and x ! 7 <= 0.250416) and
      (x ! 8 >= 0.153323 and x ! 8 <= 0.153323) and
      (x ! 9 >= -0.012064 and x ! 9 <= 0.287936) and
      (x ! 10 >= 0.107588 and x ! 10 <= 0.107588) and
      (x ! 11 >= -0.123945 and x ! 11 <= 0.176055) and
      (x ! 12 >= -0.213908 and x ! 12 <= 0.086092) and
      (x ! 13 >= -0.155773 and x ! 13 <= 0.144227) and
      (x ! 14 >= -0.142284 and x ! 14 <= 0.157716) and
      (x ! 15 >= 0.167890 and x ! 15 <= 0.167890) and
      (x ! 16 >= -0.182293 and x ! 16 <= 0.117707) and
      (x ! 17 >= 0.138031 and x ! 17 <= 0.138031) and
      (x ! 18 >= -0.100889 and x ! 18 <= 0.199111) and
      (x ! 19 >= -0.178236 and x ! 19 <= 0.121764) and
      (x ! 20 >= -0.125254 and x ! 20 <= 0.174746) and
      (x ! 21 >= -0.095307 and x ! 21 <= 0.204693) and
      (x ! 22 >= 0.085461 and x ! 22 <= 0.085461) and
      (x ! 23 >= -0.165857 and x ! 23 <= 0.134143) and
      (x ! 24 >= -0.235252 and x ! 24 <= 0.064748) and
      (x ! 25 >= -0.051104 and x ! 25 <= 0.248896) and
      (x ! 26 >= 0.216563 and x ! 26 <= 0.216563) and
      (x ! 27 >= -0.081017 and x ! 27 <= 0.218983) and
      (x ! 28 >= -0.098125 and x ! 28 <= 0.201875) and
      (x ! 29 >= -0.089500 and x ! 29 <= 0.210500))
            => isClassifiedDepression x

@property
property4 : Bool
property4 =
        forall x .
            ((x ! 0 >= 0.193511 and x ! 0 <= 0.193511) and
      (x ! 1 >= -0.151440 and x ! 1 <= 0.148560) and
      (x ! 2 >= -0.185845 and x ! 2 <= 0.114155) and
      (x ! 3 >= 0.144297 and x ! 3 <= 0.144297) and
      (x ! 4 >= -0.251845 and x ! 4 <= 0.048155) and
      (x ! 5 >= -0.128441 and x ! 5 <= -0.128441) and
      (x ! 6 >= -0.380222 and x ! 6 <= -0.080222) and
      (x ! 7 >= -0.178534 and x ! 7 <= -0.178534) and
      (x ! 8 >= 0.116889 and x ! 8 <= 0.116889) and
      (x ! 9 >= -0.146399 and x ! 9 <= 0.153601) and
      (x ! 10 >= -0.058785 and x ! 10 <= -0.058785) and
      (x ! 11 >= 0.137320 and x ! 11 <= 0.137320) and
      (x ! 12 >= -0.120606 and x ! 12 <= 0.179394) and
      (x ! 13 >= 0.258878 and x ! 13 <= 0.258878) and
      (x ! 14 >= -0.025122 and x ! 14 <= 0.274878) and
      (x ! 15 >= 0.062146 and x ! 15 <= 0.362146) and
      (x ! 16 >= -0.262384 and x ! 16 <= 0.037616) and
      (x ! 17 >= -0.125596 and x ! 17 <= 0.174404) and
      (x ! 18 >= -0.181130 and x ! 18 <= 0.118870) and
      (x ! 19 >= -0.123367 and x ! 19 <= -0.123367) and
      (x ! 20 >= -0.123031 and x ! 20 <= -0.123031) and
      (x ! 21 >= -0.012229 and x ! 21 <= 0.287771) and
      (x ! 22 >= -0.178599 and x ! 22 <= 0.121401) and
      (x ! 23 >= -0.110244 and x ! 23 <= 0.189756) and
      (x ! 24 >= -0.254611 and x ! 24 <= 0.045389) and
      (x ! 25 >= -0.018742 and x ! 25 <= 0.281258) and
      (x ! 26 >= -0.210761 and x ! 26 <= 0.089239) and
      (x ! 27 >= -0.232532 and x ! 27 <= 0.067468) and
      (x ! 28 >= -0.126350 and x ! 28 <= 0.173650) and
      (x ! 29 >= -0.175053 and x ! 29 <= 0.124947))
            => isClassifiedDepression x

@property
property17 : Bool
property17 =
        forall x .
            ((x ! 0 >= 0.238418 and x ! 0 <= 0.238418) and
      (x ! 1 >= -0.098468 and x ! 1 <= 0.201532) and
      (x ! 2 >= -0.259756 and x ! 2 <= 0.040244) and
      (x ! 3 >= -0.132108 and x ! 3 <= -0.132108) and
      (x ! 4 >= -0.263085 and x ! 4 <= 0.036915) and
      (x ! 5 >= -0.168636 and x ! 5 <= -0.168636) and
      (x ! 6 >= 0.148293 and x ! 6 <= 0.148293) and
      (x ! 7 >= 0.141697 and x ! 7 <= 0.394194) and
      (x ! 8 >= -0.057314 and x ! 8 <= 0.242686) and
      (x ! 9 >= -0.152876 and x ! 9 <= 0.147124) and
      (x ! 10 >= -0.186803 and x ! 10 <= -0.186803) and
      (x ! 11 >= -0.203098 and x ! 11 <= 0.096902) and
      (x ! 12 >= -0.174269 and x ! 12 <= 0.125731) and
      (x ! 13 >= 0.216906 and x ! 13 <= 0.216906) and
      (x ! 14 >= -0.067148 and x ! 14 <= 0.232852) and
      (x ! 15 >= 0.276898 and x ! 15 <= 0.276898) and
      (x ! 16 >= -0.106555 and x ! 16 <= -0.106555) and
      (x ! 17 >= -0.122353 and x ! 17 <= 0.177647) and
      (x ! 18 >= -0.090404 and x ! 18 <= 0.209596) and
      (x ! 19 >= -0.148252 and x ! 19 <= 0.151748) and
      (x ! 20 >= -0.149022 and x ! 20 <= 0.150978) and
      (x ! 21 >= -0.107459 and x ! 21 <= 0.192541) and
      (x ! 22 >= -0.151197 and x ! 22 <= 0.148803) and
      (x ! 23 >= -0.043960 and x ! 23 <= 0.256040) and
      (x ! 24 >= -0.052491 and x ! 24 <= -0.052491) and
      (x ! 25 >= -0.179880 and x ! 25 <= 0.120120) and
      (x ! 26 >= -0.160420 and x ! 26 <= 0.139580) and
      (x ! 27 >= -0.114945 and x ! 27 <= 0.185055) and
      (x ! 28 >= -0.160015 and x ! 28 <= 0.139985) and
      (x ! 29 >= 0.049963 and x ! 29 <= 0.049963))
            => isClassifiedDepression x

@property
property18 : Bool
property18 =
        forall x .
            ((x ! 0 >= -0.103458 and x ! 0 <= -0.103458) and
      (x ! 1 >= -0.143981 and x ! 1 <= 0.156019) and
      (x ! 2 >= -0.178033 and x ! 2 <= 0.121967) and
      (x ! 3 >= 0.146697 and x ! 3 <= 0.146697) and
      (x ! 4 >= -0.236700 and x ! 4 <= 0.063300) and
      (x ! 5 >= 0.083076 and x ! 5 <= 0.083076) and
      (x ! 6 >= -0.125443 and x ! 6 <= 0.174557) and
      (x ! 7 >= -0.166063 and x ! 7 <= 0.133937) and
      (x ! 8 >= -0.080795 and x ! 8 <= -0.080795) and
      (x ! 9 >= -0.372381 and x ! 9 <= -0.072381) and
      (x ! 10 >= 0.051835 and x ! 10 <= 0.051835) and
      (x ! 11 >= -0.256987 and x ! 11 <= -0.256987) and
      (x ! 12 >= 0.149867 and x ! 12 <= 0.149867) and
      (x ! 13 >= 0.122300 and x ! 13 <= 0.122300) and
      (x ! 14 >= -0.132417 and x ! 14 <= 0.167583) and
      (x ! 15 >= -0.081145 and x ! 15 <= 0.218855) and
      (x ! 16 >= 0.112973 and x ! 16 <= 0.112973) and
      (x ! 17 >= -0.071857 and x ! 17 <= 0.228143) and
      (x ! 18 >= -0.131362 and x ! 18 <= 0.168638) and
      (x ! 19 >= -0.187397 and x ! 19 <= 0.112603) and
      (x ! 20 >= -0.101523 and x ! 20 <= 0.198477) and
      (x ! 21 >= -0.155618 and x ! 21 <= 0.144382) and
      (x ! 22 >= 0.172606 and x ! 22 <= 0.172606) and
      (x ! 23 >= -0.109488 and x ! 23 <= 0.190512) and
      (x ! 24 >= -0.236513 and x ! 24 <= 0.063487) and
      (x ! 25 >= -0.056043 and x ! 25 <= 0.243957) and
      (x ! 26 >= -0.138661 and x ! 26 <= 0.161339) and
      (x ! 27 >= -0.139869 and x ! 27 <= 0.160131) and
      (x ! 28 >= -0.170145 and x ! 28 <= 0.129855) and
      (x ! 29 >= -0.109839 and x ! 29 <= 0.190161))
            => isClassifiedDepression x

@property
property19 : Bool
property19 =
        forall x .
            ((x ! 0 >= 0.178287 and x ! 0 <= 0.178287) and
      (x ! 1 >= -0.113553 and x ! 1 <= 0.186447) and
      (x ! 2 >= -0.161803 and x ! 2 <= 0.138197) and
      (x ! 3 >= 0.106605 and x ! 3 <= 0.106605) and
      (x ! 4 >= -0.110115 and x ! 4 <= 0.189885) and
      (x ! 5 >= 0.146656 and x ! 5 <= 0.146656) and
      (x ! 6 >= -0.182734 and x ! 6 <= 0.117266) and
      (x ! 7 >= -0.179811 and x ! 7 <= 0.120189) and
      (x ! 8 >= 0.209758 and x ! 8 <= 0.209758) and
      (x ! 9 >= -0.091060 and x ! 9 <= 0.208940) and
      (x ! 10 >= 0.001636 and x ! 10 <= 0.001636) and
      (x ! 11 >= -0.165991 and x ! 11 <= 0.134009) and
      (x ! 12 >= -0.111788 and x ! 12 <= -0.111788) and
      (x ! 13 >= 0.167366 and x ! 13 <= 0.167366) and
      (x ! 14 >= 0.279905 and x ! 14 <= 0.279905) and
      (x ! 15 >= -0.209604 and x ! 15 <= 0.090396) and
      (x ! 16 >= -0.102532 and x ! 16 <= 0.197468) and
      (x ! 17 >= -0.189893 and x ! 17 <= 0.110107) and
      (x ! 18 >= -0.227886 and x ! 18 <= 0.072114) and
      (x ! 19 >= -0.142478 and x ! 19 <= 0.157522) and
      (x ! 20 >= -0.104727 and x ! 20 <= -0.104727) and
      (x ! 21 >= -0.078495 and x ! 21 <= 0.221505) and
      (x ! 22 >= -0.226124 and x ! 22 <= 0.073876) and
      (x ! 23 >= -0.159433 and x ! 23 <= 0.140567) and
      (x ! 24 >= -0.274463 and x ! 24 <= 0.011277) and
      (x ! 25 >= -0.102432 and x ! 25 <= 0.197568) and
      (x ! 26 >= -0.249646 and x ! 26 <= 0.050354) and
      (x ! 27 >= -0.140122 and x ! 27 <= 0.159878) and
      (x ! 28 >= -0.030204 and x ! 28 <= 0.269796) and
      (x ! 29 >= 0.068793 and x ! 29 <= 0.068793))
            => isClassifiedDepression x


-- Thought - might even be interesting to just not have any bounds on the features where importance is 0, but 
-- fix all the others

-- because it might be an interesting result, that even if shap saying it isn't important, if i fix the "important" 
-- features only, and don't fix the not inportant features, then a not important feature can actually still flip the
-- classification

-- idea - use vehicle, to track down features which are "unimportant" - yet when i fix all the other features and perturb the "unimportant"
-- feature, the classification flips

-- because potentially, there could be features which have a low "importance" but for some reason have a high "impact"

-- todo also - work out how to get from arrays of upper and lower bounds in python to neater input to vehicle properties


-- want to think about the kind of table i want

-- unimportant threshold
-- the width of variation for unimportant
-- show how each changing will affect the counterexamples found