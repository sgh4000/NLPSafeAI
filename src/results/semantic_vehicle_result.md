

# semantic base results:



###### To run:


'''
*vehicle verify\\*

   *--specification src/semantic\_base.vcl*

   *--verifier Marabou*

   *--network safeNLP:src/results/base.onnx*

   *--property semantic\_i*
'''






Verifying properties:

&nbsp; semantic\_0 \[=====================================================] 1/1 queries

&nbsp;   result: ✗ - Marabou found a counterexample

&nbsp;     x: \[ 1.1227e-2, -2.439e-3, 6.0869e-2, -2.1668e-2, 0.123344, 1.2377e-2, 9.654e-2, 4.969e-2, 6.5766e-2, 0.224293, 1.265e-3, -1.1381e-2, -3.2157e-2, 0.20549, -5.753e-3, 0.115203, 0.1155, 5.6306e-2, -1.5253e-2, -5.6271e-2, -0.124234, 4.3243e-2, -0.166527, 9.0599e-2, 1.3777e-2, -6.8358e-2, -2.114e-2, -2.5952e-2, -7.4494e-2, 4.5509e-2 ]





Verifying properties:

^Zsemantic\_1 \[.....................................................] 0/1 queries

\[3]+  Stopped





Verifying properties:

&nbsp; semantic\_2 \[=====================================================] 1/1 queries

&nbsp;   result: 🗸 - Marabou proved no counterexample exists





Verifying properties:

&nbsp; semantic\_3 \[=====================================================] 1/1 queries

&nbsp;   result: 🗸 - Marabou proved no counterexample exists





Verifying properties:

^Zsemantic\_4 \[.....................................................] 0/1 queries

\[4]+  Stopped   





Verifying properties:

&nbsp; semantic\_5 \[=====================================================] 1/1 queries

&nbsp;   result: 🗸 - Marabou proved no counterexample exists





Verifying properties:

^Zsemantic\_6 \[.....................................................] 0/1 queries

\[5]+  Stopped  





Verifying properties:

&nbsp; semantic\_7 \[=====================================================] 1/1 queries

&nbsp;   result: 🗸 - Marabou proved no counterexample exists





Verifying properties:

&nbsp; semantic\_8 \[=====================================================] 1/1 queries

&nbsp;   result: 🗸 - Marabou proved no counterexample exists





Verifying properties:

&nbsp; semantic\_9 \[=====================================================] 1/1 queries

&nbsp;   result: 🗸 - Marabou proved no counterexample exists





Verifying properties:

&nbsp; semantic\_10 \[====================================================] 1/1 queries

&nbsp;   result: 🗸 - Marabou proved no counterexample exists





Verifying properties:

&nbsp; semantic\_11 \[====================================================] 1/1 queries

&nbsp;   result: ✗ - Marabou found a counterexample

&nbsp;     x: \[ 0.150976, 0.109335, 8.1252e-2, 6.8096e-2, -0.108643, 4.288e-3, 0.120345, -0.169196, -2.0159e-2, 2.9026e-2, -3.7492e-2, 5.213e-2, 2.2846e-2, 8.621e-2, 0.12741, 0.16276, -0.140713, 0.147554, 2.1391e-2, 2.0869e-2, -7.1252e-2, 9.64e-3, -4.0562e-2, 3.655e-3, 0.134567, -6.638e-3, 5.342e-2, -1.9898e-2, -1.1588e-2, 3.7276e-2 ]





Verifying properties:

&nbsp; semantic\_12 \[====================================================] 1/1 queries

&nbsp;   result: ✗ - Marabou found a counterexample

&nbsp;     x: \[ 7.6508e-2, -0.206741, 0.142623, -3.811e-3, 6.156e-3, 5.7583e-2, -3.6275e-2, -5.5562e-2, -0.102183, 6.0012e-2, -4.9112e-2, -8.7671e-2, -1.6302e-2, 1.275e-3, 7.8397e-2, 0.246307, -9.1791e-2, -4.7298e-2, 3.3854e-2, 8.739e-3, -9.1068e-2, -9.8035e-2, -5.8143e-2, 0.171477, 6.1977e-2, -0.140042, -8.089e-2, 0.124497, 6.2687e-2, -3.375e-3 ]





Verifying properties:

^Zsemantic\_13 \[....................................................] 0/1 queries

\[6]+  Stopped   





Verifying properties:

&nbsp; semantic\_14 \[====================================================] 1/1 queries

&nbsp;   result: 🗸 - Marabou proved no counterexample exists





Verifying properties:

&nbsp; semantic\_15 \[====================================================] 1/1 queries

&nbsp;   result: 🗸 - Marabou proved no counterexample exists





Verifying properties:

&nbsp; semantic\_16 \[====================================================] 1/1 queries

&nbsp;   result: 🗸 - Marabou proved no counterexample exists





Verifying properties:

^Zsemantic\_17 \[....................................................] 0/1 queries

\[7]+  Stopped    





Verifying properties:

&nbsp; semantic\_18 \[====================================================] 1/1 queries

&nbsp;   result: 🗸 - Marabou proved no counterexample exists





Verifying properties:

^Zsemantic\_19 \[....................................................] 0/1 queries

\[8]+  Stopped  





Verifying properties:

^Zsemantic\_20 \[....................................................] 0/1 queries

\[9]+  Stopped  



----------------------------------------------------------

# semantic adversarial results



###### To run:



*vehicle verify\\*

*--specification src/semantic\_adv.vcl*

*--verifier Marabou*

*--network safeNLP:src/results/adversarial.onnx*

*--property semantic\_i*





(venv\_vehicle) (base) ghofran@user:~/venv\_vehicle/projects/NLPSafeAI$ vehicle verify   --specification src/semantic\_adv.vcl   --verifier Marabou   --network safeNLP:src/results/adversarial.onnx   --property semantic\_0

Killed

(venv\_vehicle) (base) ghofran@user:~/venv\_vehicle/projects/NLPSafeAI$ vehicle verify   --specification src/semantic\_adv.vcl   --verifier Marabou   --network safeNLP:src/results/adversarial.onnx   --property semantic\_1

Killed

(venv\_vehicle) (base) ghofran@user:~/venv\_vehicle/projects/NLPSafeAI$ vehicle verify   --specification src/semantic\_adv.vcl   --verifier Marabou   --network safeNLP:src/results/adversarial.onnx   --property semantic\_2

Killed

(venv\_vehicle) (base) ghofran@user:~/venv\_vehicle/projects/NLPSafeAI$ vehicle verify   --specification src/semantic\_adv.vcl   --verifier Marabou   --network safeNLP:src/results/adversarial.onnx   --property semantic\_3

Killed

(venv\_vehicle) (base) ghofran@user:~/venv\_vehicle/projects/NLPSafeAI$ vehicle verify   --specification src/semantic\_adv.vcl   --verifier Marabou   --network safeNLP:src/results/adversarial.onnx   --property semantic\_4

Killed

(venv\_vehicle) (base) ghofran@user:~/venv\_vehicle/projects/NLPSafeAI$ vehicle verify   --specification src/semantic\_adv.vcl   --verifier Marabou   --network safeNLP:src/results/adversarial.onnx   --property semantic\_5

Killed

(venv\_vehicle) (base) ghofran@user:~/venv\_vehicle/projects/NLPSafeAI$ vehicle verify   --specification src/semantic\_adv.vcl   --verifier Marabou   --network safeNLP:src/results/adversarial.onnx   --property semantic\_6

Killed





