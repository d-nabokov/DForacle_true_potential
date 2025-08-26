This repository accompanies the paper "Unlocking the True Potential of Decryption Failure Oracles: A Hybrid Adaptive-LDPC Attack on ML-KEM Using Imperfect Oracles". It contains the code used for simulation results showed in Section 3, together with the logic to run the attack on Apple M-series processor exploiting the Data Memory-Dependent
Prefetcher (DMP). The attack was tested on MacBook Air with Apple M1.

## Installation

The code depends on the modified version of the SCA-LDPC framework available at https://github.com/d-nabokov/SCA-LDPC, which provides a belief propagation decoder working over joint distribution of few secret variables (instead of considering only the sum), plus more optimizations. Our code implicitly assumes that there is already set-up and compiled framework available at `../SCA-LDPC` path. 

Before running the code, make sure to run the following command to activate the virtual environment provided by SCA-LDPC framework
```
source ../SCA-LDPC/python-virtualenv/bin/activate
```

To run the attack on Apple M-series, the modified version of GoFetch repository is needed: https://github.com/d-nabokov/GoFetch. It provides updated `kyber_attacker` consistent with our Python logic. This repository could be installed at any location as the interaction uses sockets, follow the necessary instructions described in that repository's README file. 

## Experiments (Section 3)

The main script is `kyber_ldpc.py`, run `python kyber_ldpc.py --help` to read more about input parameters. 

As described in the paper, we first run precomputed full rotation encodings on each subset of the secret coefficients of the form `(i, i+64, i+128, i+192)`. After that we start to adaptively query new checks, using LDPC decoder after each batch. We found that running a "full" LDPC decoder each time is quite expensive and this approach has a negative effect of providing too much confidence level within each PMF, i.e., belief propagation after a lot of iterations settles on some secret key even if we don't have enough information to recover. Thus, after each batch we run a simpler version of the implementation with a few iterations. However, we still run a full decoder sometimes in order to fully enjoy error-correction benefits, this is specified by the parameter `--batch_num_for_full_ldpc`

To select a parameter set, one needs to edit `KYBER_K` value in `src/kyber_encodings.py`. To repeat the experiments that were conducted in the paper (Figure 4), for Kyber512 run
```
python kyber_ldpc.py --use_random_shuffle=1 --print_intermediate_info=0 --entropy_threshold=1.1 --batch_num_for_full_ldpc 4 7 --record_intermediate_batches=0 --max_additional_batches=18 --keys_to_test=100 --try_fix_unreliable_on_last_batch=1 --enable_full_rotation=1 --simulate_oracle=1 --seed=101 > kyber512_100.txt
```

For Kyber768, run
```
python kyber_ldpc.py --use_random_shuffle=1 --print_intermediate_info=0 --entropy_threshold=1.1 --batch_num_for_full_ldpc 4 7 --record_intermediate_batches=0 --max_additional_batches=35 --keys_to_test=100 --try_fix_unreliable_on_last_batch=1 --enable_full_rotation=1 --simulate_oracle=1 > kyber768_100.txt
```

The figures can be produced by 
```
python plot_result.py --save --prefix "kyber512" kyber512_100.txt --dist 2 4
python plot_result.py --save --prefix "kyber768" kyber768_100.txt --dist 2 4
```

### Generating the database

In `database_10000_512` and `database_10000_768` we provide a database with 10000 random balanced full rotation encodings for Kyber512 and Kyber768, resp. In `kyber_ldpc.py` we explicitly list which encodings should be taken (and what additional nega-cyclic shift should be applied to each of them) for getting good multibit encoding. To generate this database and multibit encoding, run:
```
python kyber_base_encoding.py
```

Don't forget to select correct parameter set in `src/kyber_encodings.py`. Note: this script might take a few days to run (especially for Kyber512).

## Microarchitectural Attack on Apple M1 (Section 4)

The attack uses three proccesses, one of them is `kyber_ldpc.py`, we run the following code to test 100 different secret keys for Kyber512. It will wait a connection from the modified GoFetch Kyber attacker after it is finished with a calibration phase
```
for i in {0..99}; do python kyber_ldpc.py --use_random_shuffle=1 --print_intermediate_info=0 --entropy_threshold=1.1 --batch_num_for_full_ldpc 4 7 --record_intermediate_batches=0 --max_additional_batches=18 --keys_to_test=1 --try_fix_unreliable_on_last_batch=1 --enable_full_rotation=1 --simulate_oracle=0 --base_oracle_calls=7 > out512/output_$i.txt; done
```

The victim is run as
```
./target/release/examples/kyber_victim
```

And the attacker is repeated 100 times via
```
for i in {0..99}; do ./target/release/examples/kyber_attacker 32 680 8 2; done
```
