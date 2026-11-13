# Reliability / normalization summary (83_reliability_normalization.py)

## R1 normalization: does the model ordering of the contrast survive?
```
  scheme  spearman_vs_endpoint  model_sign_flips  model_x_template_sign_flips  n_models  n_model_templates
observed                 0.982                 1                            0        29                149
z_global                 0.937                 1                            0        29                149
z_source                 0.935                 1                            0        29                149
    rank                 0.935                 2                            3        29                149
```

## R2 split-half reliability (Spearman-Brown), template-raters median per source
```
source             mode  split_half_sb
YS2008              raw         0.9460
YS2008 within_condition         0.8875
YS2009              raw         0.9560
YS2009 within_condition         0.9165
YS2011              raw         0.9475
YS2011 within_condition         0.9505
```

### model-raters
```
rater_type            who source             mode  n_items  n_raters  split_half_sb    lo    hi
    models            all YS2008              raw      192        29          0.962 0.917 0.978
    models            all YS2008 within_condition      192        29          0.785 0.685 0.849
    models     class=base YS2008              raw      192         9          0.531 0.229 0.667
    models     class=base YS2008 within_condition      192         9          0.339 0.130 0.512
    models class=instruct YS2008              raw      192        20          0.970 0.916 0.986
    models class=instruct YS2008 within_condition      192        20          0.833 0.719 0.889
    models            all YS2009              raw       96        29          0.967 0.931 0.982
    models            all YS2009 within_condition       96        29          0.864 0.797 0.915
    models     class=base YS2009              raw       96         9          0.642 0.402 0.787
    models     class=base YS2009 within_condition       96         9          0.558 0.408 0.721
    models class=instruct YS2009              raw       96        20          0.975 0.937 0.988
    models class=instruct YS2009 within_condition       96        20          0.902 0.829 0.939
    models            all YS2011              raw       10        29          0.927 0.815 0.984
    models            all YS2011 within_condition       10        29          0.912 0.781 0.982
    models     class=base YS2011              raw       10         9          0.682 0.271 0.937
    models     class=base YS2011 within_condition       10         9          0.673 0.135 0.951
    models class=instruct YS2011              raw       10        20          0.927 0.793 0.985
    models class=instruct YS2011 within_condition       10        20          0.889 0.739 0.973
```

## R3 least consistent stories (top 15 by SD across models)
```
                    story_id source   condition scenario_group  mean_z  sd_across_templates  sd_across_models  rank_inconsistent
      YS2011-Dog-intentional YS2011 intentional            Dog   0.556                0.877             1.156                  1
       YS2011-Dog-accidental YS2011  accidental            Dog   0.198                0.845             1.154                  2
    YS2008-COFFEE-accidental YS2008  accidental         COFFEE   0.358                0.498             1.063                  3
   YS2011-Allergy-accidental YS2011  accidental        Allergy   0.384                0.722             1.050                  4
   YS2011-Sibling-accidental YS2011  accidental        Sibling   0.011                0.786             1.008                  5
 YS2009-YS2009_21-accidental YS2009  accidental         SESAME   0.595                0.453             0.975                  6
YS2008-PARACHUTES-accidental YS2008  accidental     PARACHUTES   0.567                0.482             0.974                  7
     YS2008-BRIDGE-attempted YS2008   attempted         BRIDGE  -0.125                0.535             0.972                  8
   YS2011-Parent-intentional YS2011 intentional         Parent   0.144                0.708             0.955                  9
   YS2011-Poison-intentional YS2011 intentional         Poison   0.988                0.888             0.944                 10
       YS2008-RABIES-neutral YS2008     neutral         RABIES  -0.700                0.658             0.940                 11
  YS2011-Sibling-intentional YS2011 intentional        Sibling   0.094                0.706             0.936                 12
YS2008-FUMIGATION-accidental YS2008  accidental     FUMIGATION   0.197                0.523             0.927                 13
 YS2009-YS2009_08-accidental YS2009  accidental         RABIES   0.460                0.528             0.921                 14
    YS2011-Poison-accidental YS2011  accidental         Poison  -0.256                0.741             0.915                 15
```

```
            measure         factor  kruskal_H            p  mean_YS2008  mean_YS2009  mean_YS2011  mean_accidental  mean_attempted  mean_intentional  mean_neutral
sd_across_templates         source      33.74 4.720000e-08        0.491        0.477        0.775              NaN             NaN               NaN           NaN
sd_across_templates      condition      17.57 5.390000e-04          NaN          NaN          NaN            0.501           0.471             0.519         0.491
sd_across_templates scenario_group     161.01 4.470000e-13          NaN          NaN          NaN              NaN             NaN               NaN           NaN
   sd_across_models         source      31.87 1.200000e-07        0.677        0.638        0.981              NaN             NaN               NaN           NaN
   sd_across_models      condition      23.64 2.970000e-05          NaN          NaN          NaN            0.719           0.702             0.626         0.650
   sd_across_models scenario_group     128.79 1.870000e-08          NaN          NaN          NaN              NaN             NaN               NaN           NaN
```

## R4 cluster alignment (ARI)
```
   what  residualized_condition  k         factor    ARI
stories                   False  4      condition  0.385
stories                   False  4         source  0.051
stories                   False  4   intent_label  0.017
stories                   False  4  outcome_label  0.621
stories                   False  4 scenario_group  0.006
stories                    True  4      condition -0.006
stories                    True  4         source  0.054
stories                    True  4   intent_label -0.004
stories                    True  4  outcome_label -0.002
stories                    True  4 scenario_group  0.068
 models                   False  2          class  0.173
 models                   False  2         family  0.031
 models                   False  3          class  0.260
 models                   False  3         family -0.001
 models                   False  4          class  0.445
 models                   False  4         family -0.020
```

## R4 variance shares
```
                             model      cls  condition  template  scenario_group  condition:template  Residual
      HuggingFaceH4/zephyr-7b-beta instruct      0.587     0.097           0.067               0.029     0.221
                 Qwen/Qwen2.5-0.5B     base      0.001     0.656           0.163               0.005     0.175
        Qwen/Qwen2.5-0.5B-Instruct instruct      0.029     0.711           0.083               0.019     0.158
                 Qwen/Qwen2.5-1.5B     base      0.005     0.199           0.494               0.002     0.300
        Qwen/Qwen2.5-1.5B-Instruct instruct      0.374     0.347           0.046               0.108     0.125
                  Qwen/Qwen2.5-14B     base      0.341     0.259           0.056               0.146     0.198
         Qwen/Qwen2.5-14B-Instruct instruct      0.564     0.148           0.100               0.026     0.162
                   Qwen/Qwen2.5-3B     base      0.104     0.353           0.175               0.024     0.344
          Qwen/Qwen2.5-3B-Instruct instruct      0.400     0.155           0.134               0.066     0.244
                   Qwen/Qwen2.5-7B     base      0.227     0.246           0.206               0.037     0.284
          Qwen/Qwen2.5-7B-Instruct instruct      0.384     0.200           0.087               0.079     0.250
       allenai/Llama-3.1-Tulu-3-8B instruct      0.729     0.037           0.075               0.015     0.144
            allenai/OLMo-2-1124-7B     base      0.002     0.486           0.197               0.006     0.309
   allenai/OLMo-2-1124-7B-Instruct instruct      0.819     0.016           0.040               0.027     0.098
mistralai/Mistral-7B-Instruct-v0.3 instruct      0.623     0.141           0.096               0.008     0.133
         mistralai/Mistral-7B-v0.3     base      0.003     0.634           0.111               0.004     0.248
         unsloth/Meta-Llama-3.1-8B     base      0.001     0.878           0.040               0.001     0.080
unsloth/Meta-Llama-3.1-8B-Instruct instruct      0.508     0.168           0.122               0.023     0.179
                unsloth/gemma-2-9b     base      0.001     0.717           0.031               0.001     0.251
             unsloth/gemma-2-9b-it instruct      0.593     0.062           0.149               0.014     0.181
```