window.BENCHMARK_DATA = {
  "lastUpdate": 1757427772729,
  "repoUrl": "https://github.com/ami-iit/jaxsim",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "name": "Filippo Luca Ferretti",
            "username": "flferretti",
            "email": "102977828+flferretti@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "bd94e14f863b33f1d2d70c00f050196a7d5f74ef",
          "message": "Merge pull request #388 from ami-iit/benchmark_main\n\nRun performance regression checks against `main` branch",
          "timestamp": "2025-03-11T11:03:05Z",
          "url": "https://github.com/ami-iit/jaxsim/commit/bd94e14f863b33f1d2d70c00f050196a7d5f74ef"
        },
        "date": 1741691123234,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[1]",
            "value": 26.209889026381884,
            "unit": "iter/sec",
            "range": "stddev: 0.0005204304253973022",
            "extra": "mean: 38.15353811660316 msec\nrounds: 24"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[128]",
            "value": 12.8838507927293,
            "unit": "iter/sec",
            "range": "stddev: 0.00027793586015476885",
            "extra": "mean: 77.61654617766348 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[1]",
            "value": 21.070236259486062,
            "unit": "iter/sec",
            "range": "stddev: 0.00032599739047615593",
            "extra": "mean: 47.46031262700192 msec\nrounds: 22"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[128]",
            "value": 9.067266778235439,
            "unit": "iter/sec",
            "range": "stddev: 0.0006792921724309902",
            "extra": "mean: 110.28681789757684 msec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[1]",
            "value": 73.04139992361388,
            "unit": "iter/sec",
            "range": "stddev: 0.00018513514711166688",
            "extra": "mean: 13.690865742521256 msec\nrounds: 72"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[128]",
            "value": 21.599613179413872,
            "unit": "iter/sec",
            "range": "stddev: 0.0011749333312782374",
            "extra": "mean: 46.29712540190667 msec\nrounds: 22"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[1]",
            "value": 38.225699353215184,
            "unit": "iter/sec",
            "range": "stddev: 0.0005051712993463649",
            "extra": "mean: 26.160410847156665 msec\nrounds: 39"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[128]",
            "value": 38.1441549730313,
            "unit": "iter/sec",
            "range": "stddev: 0.00023508310754183205",
            "extra": "mean: 26.216336440196947 msec\nrounds: 38"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[1]",
            "value": 47.41131080771137,
            "unit": "iter/sec",
            "range": "stddev: 0.00020167524492356915",
            "extra": "mean: 21.09201333951205 msec\nrounds: 48"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[128]",
            "value": 48.61486373864245,
            "unit": "iter/sec",
            "range": "stddev: 0.00027958257728563553",
            "extra": "mean: 20.56984064330784 msec\nrounds: 49"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[1]",
            "value": 28.3640146906206,
            "unit": "iter/sec",
            "range": "stddev: 0.0002706510356398726",
            "extra": "mean: 35.2559399967692 msec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[128]",
            "value": 28.62018668341875,
            "unit": "iter/sec",
            "range": "stddev: 0.00029665361722458884",
            "extra": "mean: 34.940373068193686 msec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[1]",
            "value": 29.1376198429741,
            "unit": "iter/sec",
            "range": "stddev: 0.0001657677978624029",
            "extra": "mean: 34.31989316179949 msec\nrounds: 28"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[128]",
            "value": 14.047534481080088,
            "unit": "iter/sec",
            "range": "stddev: 0.0004656831801476427",
            "extra": "mean: 71.18686922227165 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[1]",
            "value": 4.328028060598558,
            "unit": "iter/sec",
            "range": "stddev: 0.0006571154294807941",
            "extra": "mean: 231.05210640933365 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[128]",
            "value": 2.2933524946445467,
            "unit": "iter/sec",
            "range": "stddev: 0.0005839601558344794",
            "extra": "mean: 436.042868392542 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Filippo Luca Ferretti",
            "username": "flferretti",
            "email": "102977828+flferretti@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "b5532fb160237a5817427f06c3ad993653d84ca9",
          "message": "Merge pull request #385 from ami-iit/fix/rtd_build_enum",
          "timestamp": "2025-03-11T12:02:12Z",
          "url": "https://github.com/ami-iit/jaxsim/commit/b5532fb160237a5817427f06c3ad993653d84ca9"
        },
        "date": 1742170757508,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[1]",
            "value": 26.953473867465764,
            "unit": "iter/sec",
            "range": "stddev: 0.00016960898234969532",
            "extra": "mean: 37.10096905939281 msec\nrounds: 25"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[128]",
            "value": 12.953091498003175,
            "unit": "iter/sec",
            "range": "stddev: 0.0018699083081159399",
            "extra": "mean: 77.20164720168603 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[1]",
            "value": 21.01712491313613,
            "unit": "iter/sec",
            "range": "stddev: 0.00036983505638346636",
            "extra": "mean: 47.580247257082235 msec\nrounds: 21"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[128]",
            "value": 8.980312053748797,
            "unit": "iter/sec",
            "range": "stddev: 0.00047158905982285036",
            "extra": "mean: 111.3547050497598 msec\nrounds: 9"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[1]",
            "value": 73.3289217710832,
            "unit": "iter/sec",
            "range": "stddev: 0.00007945694642804042",
            "extra": "mean: 13.637184017539225 msec\nrounds: 74"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[128]",
            "value": 21.864603863011165,
            "unit": "iter/sec",
            "range": "stddev: 0.00019420474239430254",
            "extra": "mean: 45.73602184907279 msec\nrounds: 22"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[1]",
            "value": 38.21316075501368,
            "unit": "iter/sec",
            "range": "stddev: 0.00012208957370584932",
            "extra": "mean: 26.16899466681245 msec\nrounds: 38"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[128]",
            "value": 38.19001209525572,
            "unit": "iter/sec",
            "range": "stddev: 0.00009188667786121994",
            "extra": "mean: 26.18485685486935 msec\nrounds: 38"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[1]",
            "value": 47.323959779352315,
            "unit": "iter/sec",
            "range": "stddev: 0.00026399911714963483",
            "extra": "mean: 21.130945184268057 msec\nrounds: 50"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[128]",
            "value": 48.53720388793392,
            "unit": "iter/sec",
            "range": "stddev: 0.00021084784866904976",
            "extra": "mean: 20.602752525853564 msec\nrounds: 49"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[1]",
            "value": 28.42826058125399,
            "unit": "iter/sec",
            "range": "stddev: 0.0001730031388132482",
            "extra": "mean: 35.17626402578477 msec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[128]",
            "value": 28.619011673280443,
            "unit": "iter/sec",
            "range": "stddev: 0.0003554494937614566",
            "extra": "mean: 34.94180761432896 msec\nrounds: 30"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[1]",
            "value": 29.14646393994442,
            "unit": "iter/sec",
            "range": "stddev: 0.0007403298604243962",
            "extra": "mean: 34.309479258289294 msec\nrounds: 28"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[128]",
            "value": 14.097883201818037,
            "unit": "iter/sec",
            "range": "stddev: 0.0006145402678675448",
            "extra": "mean: 70.93263475690037 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[1]",
            "value": 4.292782694593939,
            "unit": "iter/sec",
            "range": "stddev: 0.000854364504825383",
            "extra": "mean: 232.94913140125573 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[128]",
            "value": 2.327783540443568,
            "unit": "iter/sec",
            "range": "stddev: 0.00785229039610666",
            "extra": "mean: 429.59320857189596 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Filippo Luca Ferretti",
            "username": "flferretti",
            "email": "102977828+flferretti@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "eae69869dfc06f9aea18e83a8352ff25aa342b8e",
          "message": "Merge pull request #391 from ami-iit/fix/gpu_benchmarks",
          "timestamp": "2025-03-20T14:25:25Z",
          "url": "https://github.com/ami-iit/jaxsim/commit/eae69869dfc06f9aea18e83a8352ff25aa342b8e"
        },
        "date": 1742481138177,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[1]",
            "value": 30.7774968284988,
            "unit": "iter/sec",
            "range": "stddev: 0.0005705210848429037",
            "extra": "mean: 32.49127131983124 msec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[128]",
            "value": 14.146595863095165,
            "unit": "iter/sec",
            "range": "stddev: 0.00031697431912279387",
            "extra": "mean: 70.68838395311364 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[1]",
            "value": 25.68175324886898,
            "unit": "iter/sec",
            "range": "stddev: 0.000555675435981585",
            "extra": "mean: 38.93815154711214 msec\nrounds: 26"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[128]",
            "value": 10.164435370121659,
            "unit": "iter/sec",
            "range": "stddev: 0.0007605924252711689",
            "extra": "mean: 98.38224786587736 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[1]",
            "value": 69.81469839745793,
            "unit": "iter/sec",
            "range": "stddev: 0.0006163821523822769",
            "extra": "mean: 14.323631311947508 msec\nrounds: 67"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[128]",
            "value": 22.401039775722435,
            "unit": "iter/sec",
            "range": "stddev: 0.00021421729730426034",
            "extra": "mean: 44.640784981943995 msec\nrounds: 23"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[1]",
            "value": 39.54264625253923,
            "unit": "iter/sec",
            "range": "stddev: 0.00011023363754358956",
            "extra": "mean: 25.28915221337229 msec\nrounds: 40"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[128]",
            "value": 38.718691339366394,
            "unit": "iter/sec",
            "range": "stddev: 0.00012217799221209307",
            "extra": "mean: 25.827319194109013 msec\nrounds: 39"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[1]",
            "value": 48.51716698788762,
            "unit": "iter/sec",
            "range": "stddev: 0.0001257723821880265",
            "extra": "mean: 20.61126116967323 msec\nrounds: 49"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[128]",
            "value": 49.34922981894023,
            "unit": "iter/sec",
            "range": "stddev: 0.00011600846220508403",
            "extra": "mean: 20.263740764930844 msec\nrounds: 50"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[1]",
            "value": 29.534802381170334,
            "unit": "iter/sec",
            "range": "stddev: 0.0002909413439522546",
            "extra": "mean: 33.858360963252686 msec\nrounds: 30"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[128]",
            "value": 29.731915603980514,
            "unit": "iter/sec",
            "range": "stddev: 0.00016348786546905957",
            "extra": "mean: 33.633890709218875 msec\nrounds: 30"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[1]",
            "value": 27.226450796988917,
            "unit": "iter/sec",
            "range": "stddev: 0.00015933637137169617",
            "extra": "mean: 36.728988565435564 msec\nrounds: 27"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[128]",
            "value": 12.936643746085005,
            "unit": "iter/sec",
            "range": "stddev: 0.00030158212773395135",
            "extra": "mean: 77.29980199096295 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[1]",
            "value": 5.873521341538025,
            "unit": "iter/sec",
            "range": "stddev: 0.00041540292111836164",
            "extra": "mean: 170.25561700575054 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[128]",
            "value": 0.8355880800090844,
            "unit": "iter/sec",
            "range": "stddev: 0.0010119887421822474",
            "extra": "mean: 1.19676192603074 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[1]",
            "value": 5.376666178213237,
            "unit": "iter/sec",
            "range": "stddev: 0.0004388340402541893",
            "extra": "mean: 185.98885756606856 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[128]",
            "value": 3.1012058255183406,
            "unit": "iter/sec",
            "range": "stddev: 0.00098300075039445",
            "extra": "mean: 322.4552178289741 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[1]",
            "value": 4.153352438170243,
            "unit": "iter/sec",
            "range": "stddev: 0.0009839697736743166",
            "extra": "mean: 240.76935797929764 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[128]",
            "value": 2.4001936100433436,
            "unit": "iter/sec",
            "range": "stddev: 0.0015834452162262532",
            "extra": "mean: 416.63305652327836 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Filippo Luca Ferretti",
            "username": "flferretti",
            "email": "102977828+flferretti@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "eae69869dfc06f9aea18e83a8352ff25aa342b8e",
          "message": "Merge pull request #391 from ami-iit/fix/gpu_benchmarks",
          "timestamp": "2025-03-20T14:25:25Z",
          "url": "https://github.com/ami-iit/jaxsim/commit/eae69869dfc06f9aea18e83a8352ff25aa342b8e"
        },
        "date": 1742775614032,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[1]",
            "value": 31.031450319519802,
            "unit": "iter/sec",
            "range": "stddev: 0.0002925415873095915",
            "extra": "mean: 32.22537102530999 msec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[128]",
            "value": 14.011713473864212,
            "unit": "iter/sec",
            "range": "stddev: 0.0008804702548982476",
            "extra": "mean: 71.36885876700815 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[1]",
            "value": 25.594819152192827,
            "unit": "iter/sec",
            "range": "stddev: 0.0002883955561338933",
            "extra": "mean: 39.07040694656853 msec\nrounds: 26"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[128]",
            "value": 10.12648577441816,
            "unit": "iter/sec",
            "range": "stddev: 0.0006211691951562377",
            "extra": "mean: 98.75094107436863 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[1]",
            "value": 73.93022923975153,
            "unit": "iter/sec",
            "range": "stddev: 0.0002023579872324746",
            "extra": "mean: 13.526266728553715 msec\nrounds: 72"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[128]",
            "value": 22.37698085681318,
            "unit": "iter/sec",
            "range": "stddev: 0.0003513101764120844",
            "extra": "mean: 44.68878113624194 msec\nrounds: 23"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[1]",
            "value": 39.74590435431413,
            "unit": "iter/sec",
            "range": "stddev: 0.00026727278333353783",
            "extra": "mean: 25.15982504978421 msec\nrounds: 42"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[128]",
            "value": 38.81488105108883,
            "unit": "iter/sec",
            "range": "stddev: 0.0002485883509998243",
            "extra": "mean: 25.763314814330673 msec\nrounds: 39"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[1]",
            "value": 48.76070120510731,
            "unit": "iter/sec",
            "range": "stddev: 0.0001977091105011052",
            "extra": "mean: 20.50831869282589 msec\nrounds: 49"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[128]",
            "value": 48.99692317559478,
            "unit": "iter/sec",
            "range": "stddev: 0.0002195524058509135",
            "extra": "mean: 20.409444821998477 msec\nrounds: 50"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[1]",
            "value": 29.614855145284476,
            "unit": "iter/sec",
            "range": "stddev: 0.00018268937393086168",
            "extra": "mean: 33.76683745688448 msec\nrounds: 30"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[128]",
            "value": 29.465827205316206,
            "unit": "iter/sec",
            "range": "stddev: 0.00012966993329695055",
            "extra": "mean: 33.93761841580272 msec\nrounds: 30"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[1]",
            "value": 27.08931891387532,
            "unit": "iter/sec",
            "range": "stddev: 0.0002652283313929815",
            "extra": "mean: 36.91491850272373 msec\nrounds: 27"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[128]",
            "value": 13.261665698723847,
            "unit": "iter/sec",
            "range": "stddev: 0.0007915958732140992",
            "extra": "mean: 75.40530901003096 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[1]",
            "value": 5.73921683655111,
            "unit": "iter/sec",
            "range": "stddev: 0.002517212573447648",
            "extra": "mean: 174.23980108772716 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[128]",
            "value": 0.8335637942904672,
            "unit": "iter/sec",
            "range": "stddev: 0.004096109524759614",
            "extra": "mean: 1.1996682279743254 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[1]",
            "value": 5.307023055813145,
            "unit": "iter/sec",
            "range": "stddev: 0.0026827492440808726",
            "extra": "mean: 188.42955636015782 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[128]",
            "value": 3.10943110947588,
            "unit": "iter/sec",
            "range": "stddev: 0.0012082030644536869",
            "extra": "mean: 321.6022368054837 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[1]",
            "value": 4.16044469056107,
            "unit": "iter/sec",
            "range": "stddev: 0.0004432625777812931",
            "extra": "mean: 240.35892179235816 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[128]",
            "value": 2.369161170426571,
            "unit": "iter/sec",
            "range": "stddev: 0.005791933879922559",
            "extra": "mean: 422.0903214532882 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Filippo Luca Ferretti",
            "username": "flferretti",
            "email": "102977828+flferretti@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "c1846d298ac425f32fa5d9c89bafa88215855844",
          "message": "Merge pull request #394 from ami-iit/support_rev_differentiation\n\nEnable reverse-mode autodiff for contact solves via gradients override",
          "timestamp": "2025-03-28T14:33:53Z",
          "url": "https://github.com/ami-iit/jaxsim/commit/c1846d298ac425f32fa5d9c89bafa88215855844"
        },
        "date": 1743380437458,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[1]",
            "value": 30.991081071430237,
            "unit": "iter/sec",
            "range": "stddev: 0.00008949905083616838",
            "extra": "mean: 32.267348070082996 msec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[128]",
            "value": 14.588898106802722,
            "unit": "iter/sec",
            "range": "stddev: 0.0004132826332039067",
            "extra": "mean: 68.54527276009321 msec\nrounds: 15"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[1]",
            "value": 25.398718588021264,
            "unit": "iter/sec",
            "range": "stddev: 0.0001334373793605732",
            "extra": "mean: 39.37206503290396 msec\nrounds: 26"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[128]",
            "value": 10.850592873054383,
            "unit": "iter/sec",
            "range": "stddev: 0.000587032341104674",
            "extra": "mean: 92.160862701183 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[1]",
            "value": 72.85491642745387,
            "unit": "iter/sec",
            "range": "stddev: 0.00006219518580300298",
            "extra": "mean: 13.725909643939563 msec\nrounds: 70"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[128]",
            "value": 23.71144909895796,
            "unit": "iter/sec",
            "range": "stddev: 0.0002886796783127008",
            "extra": "mean: 42.17371936344231 msec\nrounds: 24"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[1]",
            "value": 39.36507019659057,
            "unit": "iter/sec",
            "range": "stddev: 0.00018050256787747787",
            "extra": "mean: 25.40323172309777 msec\nrounds: 41"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[128]",
            "value": 38.60989984344916,
            "unit": "iter/sec",
            "range": "stddev: 0.0001242533298415196",
            "extra": "mean: 25.900093086350424 msec\nrounds: 39"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[1]",
            "value": 45.21837743994232,
            "unit": "iter/sec",
            "range": "stddev: 0.009878480092735608",
            "extra": "mean: 22.11490231661164 msec\nrounds: 49"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[128]",
            "value": 49.36116278299451,
            "unit": "iter/sec",
            "range": "stddev: 0.0002323685251400584",
            "extra": "mean: 20.25884204544127 msec\nrounds: 50"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[1]",
            "value": 29.69097509802206,
            "unit": "iter/sec",
            "range": "stddev: 0.0002693257210962727",
            "extra": "mean: 33.68026805110276 msec\nrounds: 30"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[128]",
            "value": 29.81691548326578,
            "unit": "iter/sec",
            "range": "stddev: 0.0004479107304619795",
            "extra": "mean: 33.538009676461414 msec\nrounds: 30"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[1]",
            "value": 26.91720839718664,
            "unit": "iter/sec",
            "range": "stddev: 0.00012607278172998326",
            "extra": "mean: 37.15095507840697 msec\nrounds: 26"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[128]",
            "value": 13.597285831010439,
            "unit": "iter/sec",
            "range": "stddev: 0.001215557101888907",
            "extra": "mean: 73.54408905043134 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[1]",
            "value": 5.823638659160577,
            "unit": "iter/sec",
            "range": "stddev: 0.0007464639629174293",
            "extra": "mean: 171.7139504229029 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[128]",
            "value": 0.8365711003885953,
            "unit": "iter/sec",
            "range": "stddev: 0.0006854634509802923",
            "extra": "mean: 1.1953556601889432 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[1]",
            "value": 5.321780803175264,
            "unit": "iter/sec",
            "range": "stddev: 0.0012179369646974042",
            "extra": "mean: 187.90702529562017 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[128]",
            "value": 3.1919339818446453,
            "unit": "iter/sec",
            "range": "stddev: 0.0028429792873313786",
            "extra": "mean: 313.2896875962615 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[1]",
            "value": 4.085251002770874,
            "unit": "iter/sec",
            "range": "stddev: 0.0014666118641403973",
            "extra": "mean: 244.78300092741847 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[128]",
            "value": 2.479205288818282,
            "unit": "iter/sec",
            "range": "stddev: 0.0016875746342557",
            "extra": "mean: 403.3550607971847 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Filippo Luca Ferretti",
            "username": "flferretti",
            "email": "102977828+flferretti@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "b8dc8805b28278869ff415dfaedc52cdfe379d20",
          "message": "Merge pull request #396 from ami-iit/update-pixi-20250401051749\n\nUpdate `pixi` lockfile",
          "timestamp": "2025-04-01T13:21:53Z",
          "url": "https://github.com/ami-iit/jaxsim/commit/b8dc8805b28278869ff415dfaedc52cdfe379d20"
        },
        "date": 1743985210945,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[1]",
            "value": 30.43094994884611,
            "unit": "iter/sec",
            "range": "stddev: 0.0002354202407734067",
            "extra": "mean: 32.861281086557675 msec\nrounds: 28"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[128]",
            "value": 14.575836005798429,
            "unit": "iter/sec",
            "range": "stddev: 0.0006585891972580001",
            "extra": "mean: 68.60669944435358 msec\nrounds: 15"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[1]",
            "value": 25.397133741737715,
            "unit": "iter/sec",
            "range": "stddev: 0.0009491392636666373",
            "extra": "mean: 39.37452195074271 msec\nrounds: 27"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[128]",
            "value": 10.6816914285496,
            "unit": "iter/sec",
            "range": "stddev: 0.000517398924164968",
            "extra": "mean: 93.61813217401505 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[1]",
            "value": 72.03167875049594,
            "unit": "iter/sec",
            "range": "stddev: 0.00009500412749925384",
            "extra": "mean: 13.882780706302988 msec\nrounds: 72"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[128]",
            "value": 23.471106849285086,
            "unit": "iter/sec",
            "range": "stddev: 0.00020376718898495997",
            "extra": "mean: 42.605574863652386 msec\nrounds: 24"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[1]",
            "value": 38.675730942178994,
            "unit": "iter/sec",
            "range": "stddev: 0.00010568383460247941",
            "extra": "mean: 25.85600777642756 msec\nrounds: 41"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[128]",
            "value": 38.2550466367494,
            "unit": "iter/sec",
            "range": "stddev: 0.00015319005592225342",
            "extra": "mean: 26.140341939600678 msec\nrounds: 38"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[1]",
            "value": 48.19269568026804,
            "unit": "iter/sec",
            "range": "stddev: 0.00008858560640105389",
            "extra": "mean: 20.75003246621539 msec\nrounds: 48"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[128]",
            "value": 49.04617743615522,
            "unit": "iter/sec",
            "range": "stddev: 0.00007703519636250519",
            "extra": "mean: 20.388948788144152 msec\nrounds: 49"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[1]",
            "value": 29.346998764148726,
            "unit": "iter/sec",
            "range": "stddev: 0.00015369239727139257",
            "extra": "mean: 34.07503465811411 msec\nrounds: 30"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[128]",
            "value": 29.579412781307177,
            "unit": "iter/sec",
            "range": "stddev: 0.0001491117343376076",
            "extra": "mean: 33.807297237217426 msec\nrounds: 30"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[1]",
            "value": 27.180213298960158,
            "unit": "iter/sec",
            "range": "stddev: 0.00011167425098739633",
            "extra": "mean: 36.7914699197102 msec\nrounds: 26"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[128]",
            "value": 13.438532944177156,
            "unit": "iter/sec",
            "range": "stddev: 0.0007602145408073091",
            "extra": "mean: 74.41288451305948 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[1]",
            "value": 5.826319682602947,
            "unit": "iter/sec",
            "range": "stddev: 0.0005824582164795953",
            "extra": "mean: 171.63493499780694 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[128]",
            "value": 0.8342140152755144,
            "unit": "iter/sec",
            "range": "stddev: 0.0033817016292866774",
            "extra": "mean: 1.198733156826347 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[1]",
            "value": 5.166702718309451,
            "unit": "iter/sec",
            "range": "stddev: 0.00044048150996140167",
            "extra": "mean: 193.54703657639524 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[128]",
            "value": 3.1864271451250157,
            "unit": "iter/sec",
            "range": "stddev: 0.001222744051181486",
            "extra": "mean: 313.8311200775206 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[1]",
            "value": 4.16523976071087,
            "unit": "iter/sec",
            "range": "stddev: 0.0005173472764501817",
            "extra": "mean: 240.08221793919802 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[128]",
            "value": 2.4639733087509925,
            "unit": "iter/sec",
            "range": "stddev: 0.0009650346967322344",
            "extra": "mean: 405.84855219349265 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Filippo Luca Ferretti",
            "username": "flferretti",
            "email": "102977828+flferretti@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "ca7c97a66335b98b6c8cf455a9c430e12e00b683",
          "message": "Merge pull request #406 from traversaro/hotfixci\n\nUpdate tests to run fine with robot_descriptions 1.16.0",
          "timestamp": "2025-04-11T11:25:05Z",
          "url": "https://github.com/ami-iit/jaxsim/commit/ca7c97a66335b98b6c8cf455a9c430e12e00b683"
        },
        "date": 1744383607509,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[1]",
            "value": 30.225740185806888,
            "unit": "iter/sec",
            "range": "stddev: 0.0001372208580308514",
            "extra": "mean: 33.08438416570425 msec\nrounds: 28"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[128]",
            "value": 14.340358647044491,
            "unit": "iter/sec",
            "range": "stddev: 0.0005094678573219131",
            "extra": "mean: 69.73326292687229 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[1]",
            "value": 24.530000391500646,
            "unit": "iter/sec",
            "range": "stddev: 0.0003159537539681427",
            "extra": "mean: 40.76640782877803 msec\nrounds: 25"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[128]",
            "value": 10.640305976207173,
            "unit": "iter/sec",
            "range": "stddev: 0.00046810655687254843",
            "extra": "mean: 93.98225974291563 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[1]",
            "value": 71.14906931454243,
            "unit": "iter/sec",
            "range": "stddev: 0.0008374260839722691",
            "extra": "mean: 14.054997621671014 msec\nrounds: 69"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[128]",
            "value": 23.088966092430784,
            "unit": "iter/sec",
            "range": "stddev: 0.0004455134522397485",
            "extra": "mean: 43.31073102176837 msec\nrounds: 23"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[1]",
            "value": 38.02195465461988,
            "unit": "iter/sec",
            "range": "stddev: 0.0002585321189938397",
            "extra": "mean: 26.30059419837045 msec\nrounds: 38"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[128]",
            "value": 37.790221723154616,
            "unit": "iter/sec",
            "range": "stddev: 0.00016332275094950913",
            "extra": "mean: 26.461871733006678 msec\nrounds: 38"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[1]",
            "value": 46.82724496800534,
            "unit": "iter/sec",
            "range": "stddev: 0.00009909327491578557",
            "extra": "mean: 21.35508934346338 msec\nrounds: 48"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[128]",
            "value": 48.15790981134753,
            "unit": "iter/sec",
            "range": "stddev: 0.00009415000425498639",
            "extra": "mean: 20.765020822485287 msec\nrounds: 51"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[1]",
            "value": 28.508729388962088,
            "unit": "iter/sec",
            "range": "stddev: 0.0011573563850335675",
            "extra": "mean: 35.07697541887562 msec\nrounds: 30"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[128]",
            "value": 28.522605305140836,
            "unit": "iter/sec",
            "range": "stddev: 0.00015003066052786886",
            "extra": "mean: 35.05991087776834 msec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[1]",
            "value": 26.662079449499863,
            "unit": "iter/sec",
            "range": "stddev: 0.00029952690341972596",
            "extra": "mean: 37.506451883998054 msec\nrounds: 26"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[128]",
            "value": 13.372170975462517,
            "unit": "iter/sec",
            "range": "stddev: 0.0005592635026752642",
            "extra": "mean: 74.7821727552666 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[1]",
            "value": 5.6403977469631075,
            "unit": "iter/sec",
            "range": "stddev: 0.00041953383657345524",
            "extra": "mean: 177.29246142941216 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[128]",
            "value": 0.8319241973244146,
            "unit": "iter/sec",
            "range": "stddev: 0.0007055812156109495",
            "extra": "mean: 1.2020325928926467 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[1]",
            "value": 5.251020128154964,
            "unit": "iter/sec",
            "range": "stddev: 0.00034428754767202365",
            "extra": "mean: 190.43918621415892 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[128]",
            "value": 2.6457213968494355,
            "unit": "iter/sec",
            "range": "stddev: 0.0006710518503080543",
            "extra": "mean: 377.96874651685357 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[1]",
            "value": 4.115390775818093,
            "unit": "iter/sec",
            "range": "stddev: 0.0007920648390010472",
            "extra": "mean: 242.99029046669602 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[128]",
            "value": 2.1318806484671167,
            "unit": "iter/sec",
            "range": "stddev: 0.0015071935152788985",
            "extra": "mean: 469.06941095367074 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Filippo Luca Ferretti",
            "username": "flferretti",
            "email": "102977828+flferretti@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "ca7c97a66335b98b6c8cf455a9c430e12e00b683",
          "message": "Merge pull request #406 from traversaro/hotfixci\n\nUpdate tests to run fine with robot_descriptions 1.16.0",
          "timestamp": "2025-04-11T11:25:05Z",
          "url": "https://github.com/ami-iit/jaxsim/commit/ca7c97a66335b98b6c8cf455a9c430e12e00b683"
        },
        "date": 1744590058734,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[1]",
            "value": 30.381649012809312,
            "unit": "iter/sec",
            "range": "stddev: 0.00018567209796293294",
            "extra": "mean: 32.914605773320154 msec\nrounds: 28"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[128]",
            "value": 14.304098478404542,
            "unit": "iter/sec",
            "range": "stddev: 0.0006668446951127356",
            "extra": "mean: 69.91003323346376 msec\nrounds: 15"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[1]",
            "value": 24.600978371554167,
            "unit": "iter/sec",
            "range": "stddev: 0.0005518080693398801",
            "extra": "mean: 40.64878985285759 msec\nrounds: 25"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[128]",
            "value": 10.6693873359122,
            "unit": "iter/sec",
            "range": "stddev: 0.00040237565043065244",
            "extra": "mean: 93.72609396549788 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[1]",
            "value": 71.13862811058598,
            "unit": "iter/sec",
            "range": "stddev: 0.00015972290984229685",
            "extra": "mean: 14.057060510718395 msec\nrounds: 70"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[128]",
            "value": 23.468655261631245,
            "unit": "iter/sec",
            "range": "stddev: 0.0005731718154034362",
            "extra": "mean: 42.61002553626895 msec\nrounds: 25"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[1]",
            "value": 38.18592542368663,
            "unit": "iter/sec",
            "range": "stddev: 0.00012823743617544113",
            "extra": "mean: 26.18765916773363 msec\nrounds: 38"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[128]",
            "value": 37.6886560728526,
            "unit": "iter/sec",
            "range": "stddev: 0.0002091901727385859",
            "extra": "mean: 26.533182771680384 msec\nrounds: 38"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[1]",
            "value": 46.80803187913893,
            "unit": "iter/sec",
            "range": "stddev: 0.00013015634278197179",
            "extra": "mean: 21.363854873925447 msec\nrounds: 48"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[128]",
            "value": 48.0236695962584,
            "unit": "iter/sec",
            "range": "stddev: 0.00009162108239528883",
            "extra": "mean: 20.82306513448759 msec\nrounds: 49"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[1]",
            "value": 28.725487758598586,
            "unit": "iter/sec",
            "range": "stddev: 0.00016395824423768158",
            "extra": "mean: 34.81228964339043 msec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[128]",
            "value": 28.706092762917898,
            "unit": "iter/sec",
            "range": "stddev: 0.00037287086059230426",
            "extra": "mean: 34.835810232306684 msec\nrounds: 31"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[1]",
            "value": 26.76464753669989,
            "unit": "iter/sec",
            "range": "stddev: 0.0001773078457664237",
            "extra": "mean: 37.362718811401955 msec\nrounds: 26"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[128]",
            "value": 13.396705440981338,
            "unit": "iter/sec",
            "range": "stddev: 0.001174269745938199",
            "extra": "mean: 74.64521814004651 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[1]",
            "value": 5.625730947487948,
            "unit": "iter/sec",
            "range": "stddev: 0.0004402889341842559",
            "extra": "mean: 177.75467922911048 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[128]",
            "value": 0.8329535816999023,
            "unit": "iter/sec",
            "range": "stddev: 0.002521576697158161",
            "extra": "mean: 1.2005470916628838 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[1]",
            "value": 5.223067226619896,
            "unit": "iter/sec",
            "range": "stddev: 0.00040660898945048236",
            "extra": "mean: 191.45838194526732 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[128]",
            "value": 3.114182334718502,
            "unit": "iter/sec",
            "range": "stddev: 0.0023808168771253216",
            "extra": "mean: 321.11157681792974 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[1]",
            "value": 4.105883349773906,
            "unit": "iter/sec",
            "range": "stddev: 0.0005460842899310762",
            "extra": "mean: 243.55294946581125 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[128]",
            "value": 2.42057363496703,
            "unit": "iter/sec",
            "range": "stddev: 0.0015664327935971504",
            "extra": "mean: 413.1252136081457 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Filippo Luca Ferretti",
            "username": "flferretti",
            "email": "102977828+flferretti@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "020d76378aa9b605289e1ad494c8117f2ba44ac9",
          "message": "Merge pull request #400 from ami-iit/pre-commit-ci-update-config\n\n[pre-commit.ci] pre-commit autoupdate",
          "timestamp": "2025-04-14T17:19:23Z",
          "url": "https://github.com/ami-iit/jaxsim/commit/020d76378aa9b605289e1ad494c8117f2ba44ac9"
        },
        "date": 1744652629743,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[1]",
            "value": 28.88121932500741,
            "unit": "iter/sec",
            "range": "stddev: 0.008351468699089383",
            "extra": "mean: 34.62457691784948 msec\nrounds: 28"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[128]",
            "value": 14.283756882945411,
            "unit": "iter/sec",
            "range": "stddev: 0.0003670084066112773",
            "extra": "mean: 70.00959258792656 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[1]",
            "value": 24.370052430031294,
            "unit": "iter/sec",
            "range": "stddev: 0.0003395983562692417",
            "extra": "mean: 41.03396998718381 msec\nrounds: 25"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[128]",
            "value": 10.39614471112158,
            "unit": "iter/sec",
            "range": "stddev: 0.0013144432795905777",
            "extra": "mean: 96.1895036849786 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[1]",
            "value": 70.5921947886167,
            "unit": "iter/sec",
            "range": "stddev: 0.00018951053021960037",
            "extra": "mean: 14.165872062689491 msec\nrounds: 70"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[128]",
            "value": 22.80341190836492,
            "unit": "iter/sec",
            "range": "stddev: 0.0004970459217807754",
            "extra": "mean: 43.85308672309569 msec\nrounds: 23"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[1]",
            "value": 38.178620645688554,
            "unit": "iter/sec",
            "range": "stddev: 0.00025785227269814855",
            "extra": "mean: 26.19266969543931 msec\nrounds: 38"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[128]",
            "value": 37.74396719640612,
            "unit": "iter/sec",
            "range": "stddev: 0.00025070168954073",
            "extra": "mean: 26.494300262512343 msec\nrounds: 38"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[1]",
            "value": 47.49116580052769,
            "unit": "iter/sec",
            "range": "stddev: 0.0002872444622671304",
            "extra": "mean: 21.056547741956855 msec\nrounds: 49"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[128]",
            "value": 47.78609223780789,
            "unit": "iter/sec",
            "range": "stddev: 0.0002068651646236864",
            "extra": "mean: 20.926590837842348 msec\nrounds: 49"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[1]",
            "value": 28.54090931290246,
            "unit": "iter/sec",
            "range": "stddev: 0.0002175147740465033",
            "extra": "mean: 35.037426069250394 msec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[128]",
            "value": 28.373369191597988,
            "unit": "iter/sec",
            "range": "stddev: 0.0013226506164822603",
            "extra": "mean: 35.244316360431505 msec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[1]",
            "value": 23.931601148352364,
            "unit": "iter/sec",
            "range": "stddev: 0.02034038801841951",
            "extra": "mean: 41.78575406639048 msec\nrounds: 26"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[128]",
            "value": 13.137324751888562,
            "unit": "iter/sec",
            "range": "stddev: 0.0009444924380410684",
            "extra": "mean: 76.11899826532373 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[1]",
            "value": 5.623630966535506,
            "unit": "iter/sec",
            "range": "stddev: 0.00031956645366827436",
            "extra": "mean: 177.8210565292587 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[128]",
            "value": 0.8300911972021009,
            "unit": "iter/sec",
            "range": "stddev: 0.0006812364331192513",
            "extra": "mean: 1.2046869107522071 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[1]",
            "value": 5.185898158795565,
            "unit": "iter/sec",
            "range": "stddev: 0.003164216762438325",
            "extra": "mean: 192.83062824979424 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[128]",
            "value": 3.1094647386818357,
            "unit": "iter/sec",
            "range": "stddev: 0.0006732119692602518",
            "extra": "mean: 321.5987586416304 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[1]",
            "value": 4.09536352079315,
            "unit": "iter/sec",
            "range": "stddev: 0.0017750937794577932",
            "extra": "mean: 244.1785680130124 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[128]",
            "value": 2.414129352992219,
            "unit": "iter/sec",
            "range": "stddev: 0.0014109719261353496",
            "extra": "mean: 414.22801092267036 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Filippo Luca Ferretti",
            "username": "flferretti",
            "email": "102977828+flferretti@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "b1cabaf93151b7ca9aefd818fcbd67c14b1939ea",
          "message": "Merge pull request #408 from ami-iit/flferretti-patch-6",
          "timestamp": "2025-04-15T14:46:32Z",
          "url": "https://github.com/ami-iit/jaxsim/commit/b1cabaf93151b7ca9aefd818fcbd67c14b1939ea"
        },
        "date": 1745194847309,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[1]",
            "value": 30.561275647345155,
            "unit": "iter/sec",
            "range": "stddev: 0.00027313078018235655",
            "extra": "mean: 32.72114723021615 msec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[128]",
            "value": 14.360532853145429,
            "unit": "iter/sec",
            "range": "stddev: 0.00028929238880972565",
            "extra": "mean: 69.63529906767819 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[1]",
            "value": 24.50086006876817,
            "unit": "iter/sec",
            "range": "stddev: 0.0003586085042879742",
            "extra": "mean: 40.81489372998476 msec\nrounds: 25"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[128]",
            "value": 10.679550232989843,
            "unit": "iter/sec",
            "range": "stddev: 0.00019446571205655118",
            "extra": "mean: 93.63690213384953 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[1]",
            "value": 71.66912235897088,
            "unit": "iter/sec",
            "range": "stddev: 0.00009435777372335331",
            "extra": "mean: 13.953010265582375 msec\nrounds: 70"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[128]",
            "value": 23.408955487041673,
            "unit": "iter/sec",
            "range": "stddev: 0.00020042744857840571",
            "extra": "mean: 42.718693730421364 msec\nrounds: 24"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[1]",
            "value": 38.342511250165686,
            "unit": "iter/sec",
            "range": "stddev: 0.00014357047114078788",
            "extra": "mean: 26.080712175462395 msec\nrounds: 40"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[128]",
            "value": 37.92397117431305,
            "unit": "iter/sec",
            "range": "stddev: 0.00010827501044938322",
            "extra": "mean: 26.36854656922974 msec\nrounds: 38"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[1]",
            "value": 47.487731138893274,
            "unit": "iter/sec",
            "range": "stddev: 0.00033154037244057834",
            "extra": "mean: 21.05807070620358 msec\nrounds: 48"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[128]",
            "value": 47.91647914471591,
            "unit": "iter/sec",
            "range": "stddev: 0.0003097068291785",
            "extra": "mean: 20.869646890787408 msec\nrounds: 48"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[1]",
            "value": 28.77684680509697,
            "unit": "iter/sec",
            "range": "stddev: 0.0002951224672364021",
            "extra": "mean: 34.75015893064697 msec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[128]",
            "value": 28.767034520827522,
            "unit": "iter/sec",
            "range": "stddev: 0.0003607625176195273",
            "extra": "mean: 34.76201202720404 msec\nrounds: 30"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[1]",
            "value": 26.57178048495851,
            "unit": "iter/sec",
            "range": "stddev: 0.00032606455451667535",
            "extra": "mean: 37.633910176477265 msec\nrounds: 28"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[128]",
            "value": 13.46154017284391,
            "unit": "iter/sec",
            "range": "stddev: 0.0002714728364251761",
            "extra": "mean: 74.28570484210339 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[1]",
            "value": 5.48780509108675,
            "unit": "iter/sec",
            "range": "stddev: 0.00045608082928369687",
            "extra": "mean: 182.22221514830986 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[128]",
            "value": 0.8350459197242562,
            "unit": "iter/sec",
            "range": "stddev: 0.0007611160899664048",
            "extra": "mean: 1.1975389333441853 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[1]",
            "value": 5.289068178139876,
            "unit": "iter/sec",
            "range": "stddev: 0.0029005090709802958",
            "extra": "mean: 189.06922095144787 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[128]",
            "value": 2.512308347984533,
            "unit": "iter/sec",
            "range": "stddev: 0.0033292011712739806",
            "extra": "mean: 398.04031252861023 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[1]",
            "value": 4.1108638250137055,
            "unit": "iter/sec",
            "range": "stddev: 0.0009096118928250183",
            "extra": "mean: 243.25787536799908 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[128]",
            "value": 2.156233780091091,
            "unit": "iter/sec",
            "range": "stddev: 0.00039012957152484255",
            "extra": "mean: 463.7716045603156 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Filippo Luca Ferretti",
            "username": "flferretti",
            "email": "102977828+flferretti@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "a403534887b49b2144a36f5c6fe634eaa1ddd2ca",
          "message": "Merge pull request #414 from ami-iit/flferretti-patch-2\n\nExclude github-actions PRs from release notes",
          "timestamp": "2025-04-22T15:50:37Z",
          "url": "https://github.com/ami-iit/jaxsim/commit/a403534887b49b2144a36f5c6fe634eaa1ddd2ca"
        },
        "date": 1745799603910,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[1]",
            "value": 30.38762653092457,
            "unit": "iter/sec",
            "range": "stddev: 0.0004393619084292414",
            "extra": "mean: 32.90813117577084 msec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[128]",
            "value": 14.39131886901767,
            "unit": "iter/sec",
            "range": "stddev: 0.00035459810635107326",
            "extra": "mean: 69.48633472036038 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[1]",
            "value": 24.859321653992826,
            "unit": "iter/sec",
            "range": "stddev: 0.0004481290694533304",
            "extra": "mean: 40.226359106600285 msec\nrounds: 25"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[128]",
            "value": 10.672167839374374,
            "unit": "iter/sec",
            "range": "stddev: 0.0003962668342800927",
            "extra": "mean: 93.70167477225716 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[1]",
            "value": 71.67748447318976,
            "unit": "iter/sec",
            "range": "stddev: 0.00013127485682096503",
            "extra": "mean: 13.951382464796735 msec\nrounds: 69"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[128]",
            "value": 23.434464992355696,
            "unit": "iter/sec",
            "range": "stddev: 0.0008868948108260014",
            "extra": "mean: 42.67219244502485 msec\nrounds: 24"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[1]",
            "value": 38.30760554706576,
            "unit": "iter/sec",
            "range": "stddev: 0.00023780708978839258",
            "extra": "mean: 26.104476793031942 msec\nrounds: 38"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[128]",
            "value": 37.91375917955177,
            "unit": "iter/sec",
            "range": "stddev: 0.00016365367531238342",
            "extra": "mean: 26.375648884200736 msec\nrounds: 38"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[1]",
            "value": 47.65975698959308,
            "unit": "iter/sec",
            "range": "stddev: 0.00011698267725972829",
            "extra": "mean: 20.982062502298504 msec\nrounds: 48"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[128]",
            "value": 48.023005295622724,
            "unit": "iter/sec",
            "range": "stddev: 0.00010240892617791588",
            "extra": "mean: 20.82335317925531 msec\nrounds: 49"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[1]",
            "value": 28.438359120226533,
            "unit": "iter/sec",
            "range": "stddev: 0.0001231522393762939",
            "extra": "mean: 35.16377283838289 msec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[128]",
            "value": 28.943154250511462,
            "unit": "iter/sec",
            "range": "stddev: 0.00043074316255625836",
            "extra": "mean: 34.55048442007073 msec\nrounds: 31"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[1]",
            "value": 26.94388738281752,
            "unit": "iter/sec",
            "range": "stddev: 0.00020350309886809314",
            "extra": "mean: 37.11416937697392 msec\nrounds: 26"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[128]",
            "value": 13.285439024447022,
            "unit": "iter/sec",
            "range": "stddev: 0.002378127860340196",
            "extra": "mean: 75.27037670037576 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[1]",
            "value": 5.666712327038188,
            "unit": "iter/sec",
            "range": "stddev: 0.00025682455885746666",
            "extra": "mean: 176.46916629746556 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[128]",
            "value": 0.8321767283022838,
            "unit": "iter/sec",
            "range": "stddev: 0.002834746529960656",
            "extra": "mean: 1.201667826063931 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[1]",
            "value": 5.282112014372216,
            "unit": "iter/sec",
            "range": "stddev: 0.0005903545356744669",
            "extra": "mean: 189.31821159397563 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[128]",
            "value": 2.6609935895074734,
            "unit": "iter/sec",
            "range": "stddev: 0.0003894506478055625",
            "extra": "mean: 375.79947728663683 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[1]",
            "value": 4.155587520868426,
            "unit": "iter/sec",
            "range": "stddev: 0.00147775554665742",
            "extra": "mean: 240.63986018300056 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[128]",
            "value": 2.1062800492021823,
            "unit": "iter/sec",
            "range": "stddev: 0.0007400020878167491",
            "extra": "mean: 474.77067466825247 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Filippo Luca Ferretti",
            "username": "flferretti",
            "email": "102977828+flferretti@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "a403534887b49b2144a36f5c6fe634eaa1ddd2ca",
          "message": "Merge pull request #414 from ami-iit/flferretti-patch-2\n\nExclude github-actions PRs from release notes",
          "timestamp": "2025-04-22T15:50:37Z",
          "url": "https://github.com/ami-iit/jaxsim/commit/a403534887b49b2144a36f5c6fe634eaa1ddd2ca"
        },
        "date": 1746404439739,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[1]",
            "value": 30.531399238836723,
            "unit": "iter/sec",
            "range": "stddev: 0.0005570122453086252",
            "extra": "mean: 32.753166410007644 msec\nrounds: 28"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[128]",
            "value": 14.285834442542306,
            "unit": "iter/sec",
            "range": "stddev: 0.00024492804724007036",
            "extra": "mean: 69.99941123649478 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[1]",
            "value": 24.52690328183461,
            "unit": "iter/sec",
            "range": "stddev: 0.00015562818178871878",
            "extra": "mean: 40.771555565297604 msec\nrounds: 25"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[128]",
            "value": 10.480228646226351,
            "unit": "iter/sec",
            "range": "stddev: 0.0005497581559477828",
            "extra": "mean: 95.41776556182992 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[1]",
            "value": 71.42970998126721,
            "unit": "iter/sec",
            "range": "stddev: 0.00010429747559632141",
            "extra": "mean: 13.999776847228624 msec\nrounds: 69"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[128]",
            "value": 23.08628168674072,
            "unit": "iter/sec",
            "range": "stddev: 0.00018695640161131308",
            "extra": "mean: 43.31576706760604 msec\nrounds: 23"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[1]",
            "value": 38.27304973202589,
            "unit": "iter/sec",
            "range": "stddev: 0.00045952138924607424",
            "extra": "mean: 26.12804589656795 msec\nrounds: 41"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[128]",
            "value": 37.83713690449094,
            "unit": "iter/sec",
            "range": "stddev: 0.00008038974775289302",
            "extra": "mean: 26.42906101812658 msec\nrounds: 38"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[1]",
            "value": 47.49128426201124,
            "unit": "iter/sec",
            "range": "stddev: 0.00008188570271964568",
            "extra": "mean: 21.056495218848188 msec\nrounds: 48"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[128]",
            "value": 47.41177632426973,
            "unit": "iter/sec",
            "range": "stddev: 0.0002868482073891334",
            "extra": "mean: 21.091806245785133 msec\nrounds: 51"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[1]",
            "value": 28.659785185647305,
            "unit": "iter/sec",
            "range": "stddev: 0.0002235787082743474",
            "extra": "mean: 34.89209683611989 msec\nrounds: 30"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[128]",
            "value": 28.778255680447874,
            "unit": "iter/sec",
            "range": "stddev: 0.00012326655637259198",
            "extra": "mean: 34.74845769333428 msec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[1]",
            "value": 26.443235941967846,
            "unit": "iter/sec",
            "range": "stddev: 0.0001850677292036907",
            "extra": "mean: 37.816854268312454 msec\nrounds: 25"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[128]",
            "value": 13.248119469042402,
            "unit": "iter/sec",
            "range": "stddev: 0.0006727911775389534",
            "extra": "mean: 75.48241109515612 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[1]",
            "value": 5.6518155899934905,
            "unit": "iter/sec",
            "range": "stddev: 0.0006102156662927648",
            "extra": "mean: 176.93429378171763 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[128]",
            "value": 0.8322366028765941,
            "unit": "iter/sec",
            "range": "stddev: 0.00118063585258768",
            "extra": "mean: 1.2015813730657101 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[1]",
            "value": 5.268984141682936,
            "unit": "iter/sec",
            "range": "stddev: 0.0007836280040137856",
            "extra": "mean: 189.78990505759916 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[128]",
            "value": 2.6440897166371586,
            "unit": "iter/sec",
            "range": "stddev: 0.0008828181786613415",
            "extra": "mean: 378.2019928097725 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[1]",
            "value": 4.028285577288715,
            "unit": "iter/sec",
            "range": "stddev: 0.0015115290413685223",
            "extra": "mean: 248.24456479400396 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[128]",
            "value": 2.142809055190191,
            "unit": "iter/sec",
            "range": "stddev: 0.0011823171955834433",
            "extra": "mean: 466.67713932693005 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Filippo Luca Ferretti",
            "username": "flferretti",
            "email": "102977828+flferretti@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "e5bd01e2d1340509b10f4651d108168fc8c91ae9",
          "message": "Merge pull request #419 from ami-iit/flferretti-patch-2\n\nUse Pixi Docker image for GPU benchmarking",
          "timestamp": "2025-05-13T13:47:09Z",
          "url": "https://github.com/ami-iit/jaxsim/commit/e5bd01e2d1340509b10f4651d108168fc8c91ae9"
        },
        "date": 1747145325876,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[1]",
            "value": 31.519731083593296,
            "unit": "iter/sec",
            "range": "stddev: 0.0009189433097806864",
            "extra": "mean: 31.726159000148378 msec\nrounds: 30"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[128]",
            "value": 13.950833135595115,
            "unit": "iter/sec",
            "range": "stddev: 0.0004689243579389566",
            "extra": "mean: 71.68030685196365 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[1]",
            "value": 26.80041604252841,
            "unit": "iter/sec",
            "range": "stddev: 0.00025812697292321906",
            "extra": "mean: 37.31285359201677 msec\nrounds: 27"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[128]",
            "value": 9.488666664571301,
            "unit": "iter/sec",
            "range": "stddev: 0.0004552901860351288",
            "extra": "mean: 105.38888500886969 msec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[1]",
            "value": 77.79981396903197,
            "unit": "iter/sec",
            "range": "stddev: 0.00024187761490704321",
            "extra": "mean: 12.853501171584389 msec\nrounds: 76"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[128]",
            "value": 20.903184229249334,
            "unit": "iter/sec",
            "range": "stddev: 0.0003835096612513719",
            "extra": "mean: 47.83960132737688 msec\nrounds: 21"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[1]",
            "value": 39.76884528053919,
            "unit": "iter/sec",
            "range": "stddev: 0.0004901743506371821",
            "extra": "mean: 25.145311435264333 msec\nrounds: 41"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[128]",
            "value": 39.96008698485469,
            "unit": "iter/sec",
            "range": "stddev: 0.00016647723099480743",
            "extra": "mean: 25.02497055071504 msec\nrounds: 38"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[1]",
            "value": 52.1655825383572,
            "unit": "iter/sec",
            "range": "stddev: 0.00011512780154074382",
            "extra": "mean: 19.169727458994693 msec\nrounds: 52"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[128]",
            "value": 52.41708690048107,
            "unit": "iter/sec",
            "range": "stddev: 0.0004114946190480842",
            "extra": "mean: 19.077748481112604 msec\nrounds: 52"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[1]",
            "value": 31.510899160897576,
            "unit": "iter/sec",
            "range": "stddev: 0.0002454307910923614",
            "extra": "mean: 31.735051256198915 msec\nrounds: 31"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[128]",
            "value": 31.797064553209765,
            "unit": "iter/sec",
            "range": "stddev: 0.00025266893147107356",
            "extra": "mean: 31.4494439675896 msec\nrounds: 33"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[1]",
            "value": 28.414005647505956,
            "unit": "iter/sec",
            "range": "stddev: 0.00023964217848563183",
            "extra": "mean: 35.19391149581809 msec\nrounds: 28"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[128]",
            "value": 12.867284636826264,
            "unit": "iter/sec",
            "range": "stddev: 0.0005370582411932912",
            "extra": "mean: 77.71647462728791 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[1]",
            "value": 5.957812834715543,
            "unit": "iter/sec",
            "range": "stddev: 0.0013558226225032107",
            "extra": "mean: 167.84683032892644 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[128]",
            "value": 0.8287469262990353,
            "unit": "iter/sec",
            "range": "stddev: 0.0024894357313183534",
            "extra": "mean: 1.2066409759921952 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[1]",
            "value": 5.607929305952749,
            "unit": "iter/sec",
            "range": "stddev: 0.0010140575528107703",
            "extra": "mean: 178.31893831801912 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[128]",
            "value": 3.0698508981192223,
            "unit": "iter/sec",
            "range": "stddev: 0.007273382082228479",
            "extra": "mean: 325.7487197872251 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[1]",
            "value": 4.529481698540647,
            "unit": "iter/sec",
            "range": "stddev: 0.00033558511200268225",
            "extra": "mean: 220.77581201447174 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[128]",
            "value": 2.449584006765416,
            "unit": "iter/sec",
            "range": "stddev: 0.0002185614639587191",
            "extra": "mean: 408.23258040472865 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Filippo Luca Ferretti",
            "username": "flferretti",
            "email": "102977828+flferretti@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "6b5371806c9050791331d610a555216bd0c46791",
          "message": "Merge pull request #425 from ami-iit/flferretti-patch-2",
          "timestamp": "2025-05-15T13:15:55Z",
          "url": "https://github.com/ami-iit/jaxsim/commit/6b5371806c9050791331d610a555216bd0c46791"
        },
        "date": 1747614031884,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[1]",
            "value": 31.563892779553804,
            "unit": "iter/sec",
            "range": "stddev: 0.0001011353279731985",
            "extra": "mean: 31.681770274158694 msec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[128]",
            "value": 14.044859675706089,
            "unit": "iter/sec",
            "range": "stddev: 0.00040768438407519516",
            "extra": "mean: 71.20042656814414 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[1]",
            "value": 26.886665849655746,
            "unit": "iter/sec",
            "range": "stddev: 0.0001981173168298998",
            "extra": "mean: 37.19315758940799 msec\nrounds: 27"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[128]",
            "value": 9.708344940422439,
            "unit": "iter/sec",
            "range": "stddev: 0.00037656934906699427",
            "extra": "mean: 103.0041686957702 msec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[1]",
            "value": 77.01808913658277,
            "unit": "iter/sec",
            "range": "stddev: 0.0007565222613482701",
            "extra": "mean: 12.983962744474411 msec\nrounds: 75"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[128]",
            "value": 20.86038586504797,
            "unit": "iter/sec",
            "range": "stddev: 0.00029753629881867737",
            "extra": "mean: 47.93775179756007 msec\nrounds: 21"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[1]",
            "value": 40.129326809857986,
            "unit": "iter/sec",
            "range": "stddev: 0.00014016892457098241",
            "extra": "mean: 24.919431236368126 msec\nrounds: 41"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[128]",
            "value": 40.00404184543124,
            "unit": "iter/sec",
            "range": "stddev: 0.00037937419042323797",
            "extra": "mean: 24.997474101837724 msec\nrounds: 40"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[1]",
            "value": 52.03173014163487,
            "unit": "iter/sec",
            "range": "stddev: 0.00023195450245980755",
            "extra": "mean: 19.21904186691301 msec\nrounds: 52"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[128]",
            "value": 52.614453003860774,
            "unit": "iter/sec",
            "range": "stddev: 0.00021063224499102244",
            "extra": "mean: 19.006184477991653 msec\nrounds: 53"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[1]",
            "value": 31.387768545095437,
            "unit": "iter/sec",
            "range": "stddev: 0.0003275982792065585",
            "extra": "mean: 31.85954422224313 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[128]",
            "value": 31.605720642331963,
            "unit": "iter/sec",
            "range": "stddev: 0.00017211895316631028",
            "extra": "mean: 31.639841765247507 msec\nrounds: 34"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[1]",
            "value": 28.161763278629266,
            "unit": "iter/sec",
            "range": "stddev: 0.0005168928230619798",
            "extra": "mean: 35.509140180822996 msec\nrounds: 28"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[128]",
            "value": 12.69519221621364,
            "unit": "iter/sec",
            "range": "stddev: 0.0003466938078650083",
            "extra": "mean: 78.7699770880863 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[1]",
            "value": 6.112648280374618,
            "unit": "iter/sec",
            "range": "stddev: 0.0007018328531641755",
            "extra": "mean: 163.59521342175347 msec\nrounds: 7"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[128]",
            "value": 0.8302386078421997,
            "unit": "iter/sec",
            "range": "stddev: 0.002772698952372499",
            "extra": "mean: 1.204473016015254 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[1]",
            "value": 5.553146668216488,
            "unit": "iter/sec",
            "range": "stddev: 0.00031899180185154547",
            "extra": "mean: 180.0780818060351 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[128]",
            "value": 3.114258020721069,
            "unit": "iter/sec",
            "range": "stddev: 0.0001996929463858718",
            "extra": "mean: 321.1037728236988 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[1]",
            "value": 4.420437581581546,
            "unit": "iter/sec",
            "range": "stddev: 0.000646530857013627",
            "extra": "mean: 226.22194783762097 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[128]",
            "value": 2.4453877940007804,
            "unit": "iter/sec",
            "range": "stddev: 0.0016106511396445193",
            "extra": "mean: 408.9330953778699 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Filippo Luca Ferretti",
            "username": "flferretti",
            "email": "102977828+flferretti@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "ae646ec01f5bf54d36b4b138b9b52f47a0accacc",
          "message": "Merge pull request #428 from ami-iit/flferretti-patch-2\n\nIncrease minimum tolerance for hardware parametrization when using X32",
          "timestamp": "2025-05-19T13:13:10Z",
          "url": "https://github.com/ami-iit/jaxsim/commit/ae646ec01f5bf54d36b4b138b9b52f47a0accacc"
        },
        "date": 1747736454517,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[1]",
            "value": 31.965130018317236,
            "unit": "iter/sec",
            "range": "stddev: 0.00028283631181859196",
            "extra": "mean: 31.284089863765985 msec\nrounds: 30"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[128]",
            "value": 14.122384000754241,
            "unit": "iter/sec",
            "range": "stddev: 0.0004579433748366867",
            "extra": "mean: 70.80957435703438 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[1]",
            "value": 27.06160726080346,
            "unit": "iter/sec",
            "range": "stddev: 0.00014982028547945374",
            "extra": "mean: 36.95272015304201 msec\nrounds: 27"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[128]",
            "value": 9.769018744330989,
            "unit": "iter/sec",
            "range": "stddev: 0.0002981934185509951",
            "extra": "mean: 102.36442637396976 msec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[1]",
            "value": 77.93742844536521,
            "unit": "iter/sec",
            "range": "stddev: 0.00016918833337074902",
            "extra": "mean: 12.83080568537116 msec\nrounds: 75"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[128]",
            "value": 21.00429930128478,
            "unit": "iter/sec",
            "range": "stddev: 0.0008055652327672161",
            "extra": "mean: 47.60930063202977 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[1]",
            "value": 40.355606921970036,
            "unit": "iter/sec",
            "range": "stddev: 0.00033699597622436175",
            "extra": "mean: 24.77970414206778 msec\nrounds: 38"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[128]",
            "value": 39.79534607495362,
            "unit": "iter/sec",
            "range": "stddev: 0.00010651088169783933",
            "extra": "mean: 25.12856649409514 msec\nrounds: 40"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[1]",
            "value": 52.025880377381455,
            "unit": "iter/sec",
            "range": "stddev: 0.00008677556683568609",
            "extra": "mean: 19.221202846473226 msec\nrounds: 52"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[128]",
            "value": 52.74315408666106,
            "unit": "iter/sec",
            "range": "stddev: 0.00005852988991471834",
            "extra": "mean: 18.959806581853698 msec\nrounds: 53"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[1]",
            "value": 31.468071923420744,
            "unit": "iter/sec",
            "range": "stddev: 0.0004879294670966327",
            "extra": "mean: 31.77824184568899 msec\nrounds: 31"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[128]",
            "value": 31.743705955524508,
            "unit": "iter/sec",
            "range": "stddev: 0.000275041616851517",
            "extra": "mean: 31.502307934715645 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[1]",
            "value": 28.47648243213289,
            "unit": "iter/sec",
            "range": "stddev: 0.00013788607208073457",
            "extra": "mean: 35.11669681756757 msec\nrounds: 28"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[128]",
            "value": 12.815226236868297,
            "unit": "iter/sec",
            "range": "stddev: 0.0007771365243936664",
            "extra": "mean: 78.03217684312796 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[1]",
            "value": 6.022636447766811,
            "unit": "iter/sec",
            "range": "stddev: 0.003081953914491487",
            "extra": "mean: 166.04023979743943 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[128]",
            "value": 0.8312252817129976,
            "unit": "iter/sec",
            "range": "stddev: 0.002906359216535415",
            "extra": "mean: 1.2030432928353547 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[1]",
            "value": 5.5592178559445395,
            "unit": "iter/sec",
            "range": "stddev: 0.0008868321171207182",
            "extra": "mean: 179.88141963724047 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[128]",
            "value": 3.11902440531516,
            "unit": "iter/sec",
            "range": "stddev: 0.002779354775403615",
            "extra": "mean: 320.6130732083693 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[1]",
            "value": 4.507154089318723,
            "unit": "iter/sec",
            "range": "stddev: 0.002832980787081811",
            "extra": "mean: 221.86949462629855 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[128]",
            "value": 2.441517410648027,
            "unit": "iter/sec",
            "range": "stddev: 0.004014142061926503",
            "extra": "mean: 409.58135118708014 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Filippo Luca Ferretti",
            "username": "flferretti",
            "email": "102977828+flferretti@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "9d10093e0ed56f3043b915a51b8f484ed978aeb2",
          "message": "Merge pull request #431 from ami-iit/update-pixi-20250520104719\n\nUpdate `pixi` lockfile",
          "timestamp": "2025-05-20T12:37:46Z",
          "url": "https://github.com/ami-iit/jaxsim/commit/9d10093e0ed56f3043b915a51b8f484ed978aeb2"
        },
        "date": 1747753460407,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[1]",
            "value": 31.32756725199066,
            "unit": "iter/sec",
            "range": "stddev: 0.011701503589121844",
            "extra": "mean: 31.920767800329486 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[128]",
            "value": 14.897502072663855,
            "unit": "iter/sec",
            "range": "stddev: 0.0006544026178173679",
            "extra": "mean: 67.12534726442149 msec\nrounds: 15"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[1]",
            "value": 27.66660570910434,
            "unit": "iter/sec",
            "range": "stddev: 0.0002577399735469302",
            "extra": "mean: 36.14465795024963 msec\nrounds: 28"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[128]",
            "value": 10.588987965163476,
            "unit": "iter/sec",
            "range": "stddev: 0.0003866384615333466",
            "extra": "mean: 94.43773128176954 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[1]",
            "value": 79.83370397451029,
            "unit": "iter/sec",
            "range": "stddev: 0.0001689162221805986",
            "extra": "mean: 12.526037878929994 msec\nrounds: 78"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[128]",
            "value": 22.76074062719521,
            "unit": "iter/sec",
            "range": "stddev: 0.00041352427714563047",
            "extra": "mean: 43.935301420076385 msec\nrounds: 23"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[1]",
            "value": 41.580806983575386,
            "unit": "iter/sec",
            "range": "stddev: 0.00013074722204989335",
            "extra": "mean: 24.049557296831797 msec\nrounds: 41"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[128]",
            "value": 40.716360360198195,
            "unit": "iter/sec",
            "range": "stddev: 0.00026092691787953245",
            "extra": "mean: 24.560152016375667 msec\nrounds: 41"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[1]",
            "value": 52.759351861797114,
            "unit": "iter/sec",
            "range": "stddev: 0.00008988014472013515",
            "extra": "mean: 18.953985686167936 msec\nrounds: 53"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[128]",
            "value": 52.28012296959551,
            "unit": "iter/sec",
            "range": "stddev: 0.0005071965825910952",
            "extra": "mean: 19.127728536169833 msec\nrounds: 52"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[1]",
            "value": 32.205111362803365,
            "unit": "iter/sec",
            "range": "stddev: 0.0002499798243715592",
            "extra": "mean: 31.050971652748004 msec\nrounds: 33"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[128]",
            "value": 32.61564198536606,
            "unit": "iter/sec",
            "range": "stddev: 0.0002930786772196211",
            "extra": "mean: 30.660135417499326 msec\nrounds: 33"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[1]",
            "value": 29.383407402282838,
            "unit": "iter/sec",
            "range": "stddev: 0.00023886697384724765",
            "extra": "mean: 34.032812679250696 msec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[128]",
            "value": 13.713402641972749,
            "unit": "iter/sec",
            "range": "stddev: 0.0007858985978682111",
            "extra": "mean: 72.92136212345213 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[1]",
            "value": 6.304297132503958,
            "unit": "iter/sec",
            "range": "stddev: 0.00041581982000824745",
            "extra": "mean: 158.62196514249914 msec\nrounds: 7"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[128]",
            "value": 0.8389857600958205,
            "unit": "iter/sec",
            "range": "stddev: 0.0007515819140007107",
            "extra": "mean: 1.1919153429800644 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[1]",
            "value": 5.826869529530658,
            "unit": "iter/sec",
            "range": "stddev: 0.0002737748331495827",
            "extra": "mean: 171.6187388325731 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[128]",
            "value": 3.2947090527530247,
            "unit": "iter/sec",
            "range": "stddev: 0.0007466611610651291",
            "extra": "mean: 303.516937000677 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[1]",
            "value": 4.675432324794508,
            "unit": "iter/sec",
            "range": "stddev: 0.001107153150119978",
            "extra": "mean: 213.88396420516074 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[128]",
            "value": 2.594003559525794,
            "unit": "iter/sec",
            "range": "stddev: 0.0016765323089330464",
            "extra": "mean: 385.5044825701043 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Alessandro Croci",
            "username": "xela-95",
            "email": "57228872+xela-95@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "a72309f4d236f566e6891045a44d5bf07638f0b2",
          "message": "Add kinematic constraints support (#399)\n\n* WIP Try adding kinematic constraint in relaxed contact model\n\n* Add inertial transformation for kinematic constraint forces in RelaxedRigidContacts\n\n* Add kinematic constraint forces to link contact forces computation\n\n* Add logarithm map computation for SO(3) rotation matrix\n\n* Implement Baumgarte stabilization for kinematic constraints in RelaxedRigidContacts\n\n* Fix constraint forces\n\n* Refactor link_contact_forces to separate kinematic constraint forces for F1 and F2\n\n* Rename log_SO3 to log_vee for clarity in SO(3) logarithm map computation\n\n* Refactor RelaxedRigidContacts to improve kinematic constraint handling and adding Baumgarte terms\n\n* Working implementation\n\n* Fix kinematic constraint generalization\n\n* WIP Update `mujoco.loaders` to allow visualizing frames\n\n* Refactor RodModelToMjcf to improve base link handling\n\n* Manage cases when no kin. constraint defined\n\n* Fix vel.repr. in RelaxedRigidContacts constraints\n\n* Fix type conversion for roll, pitch, yaw and improve warning message formatting IN `mujoco.loaders`\n\n* Add fixture for double pendulum model and utility function to load models from file\n\n* Add test asset: SDF model for double pendulum\n\n* Add test for simulation with kinematic constraints using double pendulum model\n\n* Add cartpole model and update simulation tests for kinematic constraints\n\n* Add fixture for JaxSim cartpole model\n\n* Move kinematic constraints functions into a separate rbda module\n\n* Refactor contact model to process kinematic constraint wrenches when available\n\n* Move ConstraintType and ConstraintMap classes to kinematic_constraints.py\n\n* Refactor kinematic constraints to be a batchable dataclass\n\nIntegrate also the Baumgarte coefficients for each constraint.\n\n* Comment out Connect constraint type\n\n* Minor changes\n\n* Refacor type annotations in `model.py` to avoid circular import errors\n\n* Fix circular import issue\n\nMoved ConstraintMap and ConstraintType in `src/jaxsim/api/kin_dyn_parameters.py`\n\n* Fix warning message in RodModelToMjcf\n\n* Formatting\n\n* Fix parameter name in RelaxedRigidContacts\n\n* Remove constraints from hash calculation in KinDynParameters\n\n* Maintain constraints in model reduction\n\n* Update docstring in `kinematic_constraints.py`\n\n* Update cartpole model in unit tests\n\n* Address review comments\n\n* Clarify constraints usage in documentation and enforce Relaxed-Rigid contact model requirement for constraints\n\n* Formatting and minor refactors\n\n* Reduce simulation time in unit tests for kin constraints\n\n* Optimize kin constraint wrench pair handling\n\n* Use jaxlie instead of custom functions in `log_vee` method\n\n* Fix contact model check in JaxSimModel\n\n* Formatting\n\n* Update examples/assets/cartpole.urdf\n\nCo-authored-by: Filippo Luca Ferretti <102977828+flferretti@users.noreply.github.com>\n\n* Fix check for kinematic constraints support\n\n---------\n\nCo-authored-by: Filippo Luca Ferretti <filippoluca.ferretti@outlook.com>\nCo-authored-by: Carlotta Sartore <carlotta.sartore@iit.it>\nCo-authored-by: Omar Younis <omar.younis@iit.it>",
          "timestamp": "2025-05-23T12:13:53Z",
          "url": "https://github.com/ami-iit/jaxsim/commit/a72309f4d236f566e6891045a44d5bf07638f0b2"
        },
        "date": 1748218815521,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[1]",
            "value": 33.993431464815046,
            "unit": "iter/sec",
            "range": "stddev: 0.00011379114690042765",
            "extra": "mean: 29.41744792769896 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[128]",
            "value": 15.335758186894385,
            "unit": "iter/sec",
            "range": "stddev: 0.0002569971393237116",
            "extra": "mean: 65.20707928575575 msec\nrounds: 15"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[1]",
            "value": 27.81227727717253,
            "unit": "iter/sec",
            "range": "stddev: 0.0003333144344285249",
            "extra": "mean: 35.95534411059426 msec\nrounds: 28"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[128]",
            "value": 10.712014109378831,
            "unit": "iter/sec",
            "range": "stddev: 0.000355638543251366",
            "extra": "mean: 93.35312573239207 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[1]",
            "value": 80.90913740683327,
            "unit": "iter/sec",
            "range": "stddev: 0.00007716400895635888",
            "extra": "mean: 12.359543458876919 msec\nrounds: 79"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[128]",
            "value": 23.09764270724867,
            "unit": "iter/sec",
            "range": "stddev: 0.0002276081297102437",
            "extra": "mean: 43.294461373158775 msec\nrounds: 23"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[1]",
            "value": 41.90213946379174,
            "unit": "iter/sec",
            "range": "stddev: 0.00036971917208085696",
            "extra": "mean: 23.8651298667963 msec\nrounds: 41"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[128]",
            "value": 40.44415641656463,
            "unit": "iter/sec",
            "range": "stddev: 0.00035081868281895353",
            "extra": "mean: 24.725450809265293 msec\nrounds: 43"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[1]",
            "value": 54.67012385426936,
            "unit": "iter/sec",
            "range": "stddev: 0.000075157626856897",
            "extra": "mean: 18.291526148095727 msec\nrounds: 55"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[128]",
            "value": 54.34914365568856,
            "unit": "iter/sec",
            "range": "stddev: 0.00010303892075823124",
            "extra": "mean: 18.39955393474416 msec\nrounds: 56"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[1]",
            "value": 32.36723642626764,
            "unit": "iter/sec",
            "range": "stddev: 0.00026655014681168765",
            "extra": "mean: 30.895439660966847 msec\nrounds: 33"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[128]",
            "value": 32.236570985289006,
            "unit": "iter/sec",
            "range": "stddev: 0.001513562067250962",
            "extra": "mean: 31.020669054917313 msec\nrounds: 33"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[1]",
            "value": 29.87140024663237,
            "unit": "iter/sec",
            "range": "stddev: 0.00013923340967226443",
            "extra": "mean: 33.47683709981884 msec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[128]",
            "value": 13.854669663385026,
            "unit": "iter/sec",
            "range": "stddev: 0.0006566164035445914",
            "extra": "mean: 72.1778306012441 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[1]",
            "value": 6.353835509636873,
            "unit": "iter/sec",
            "range": "stddev: 0.001935883998658022",
            "extra": "mean: 157.38525155133433 msec\nrounds: 7"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[128]",
            "value": 0.8410626112721016,
            "unit": "iter/sec",
            "range": "stddev: 0.0036855911587465916",
            "extra": "mean: 1.1889721247833223 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[1]",
            "value": 5.901340276148056,
            "unit": "iter/sec",
            "range": "stddev: 0.003998411243567624",
            "extra": "mean: 169.45303154976241 msec\nrounds: 7"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[128]",
            "value": 3.311309887969375,
            "unit": "iter/sec",
            "range": "stddev: 0.0038031848094438",
            "extra": "mean: 301.9952930510044 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[1]",
            "value": 4.787965200446324,
            "unit": "iter/sec",
            "range": "stddev: 0.0006012239952859352",
            "extra": "mean: 208.8569900020957 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[128]",
            "value": 2.554482824207584,
            "unit": "iter/sec",
            "range": "stddev: 0.0014927054231157367",
            "extra": "mean: 391.46867245435715 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Filippo Luca Ferretti",
            "username": "flferretti",
            "email": "102977828+flferretti@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "ad48aa2e6a8b6daf3621ed1b5bd46e0aff149b9d",
          "message": "Merge pull request #436 from ami-iit/flferretti-patch-2",
          "timestamp": "2025-05-27T11:47:27Z",
          "url": "https://github.com/ami-iit/jaxsim/commit/ad48aa2e6a8b6daf3621ed1b5bd46e0aff149b9d"
        },
        "date": 1748823637548,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[1]",
            "value": 31.90350871875609,
            "unit": "iter/sec",
            "range": "stddev: 0.011497602080915756",
            "extra": "mean: 31.34451476216782 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[128]",
            "value": 15.210885880097793,
            "unit": "iter/sec",
            "range": "stddev: 0.0020373237637327335",
            "extra": "mean: 65.7423905407389 msec\nrounds: 15"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[1]",
            "value": 27.87145581986009,
            "unit": "iter/sec",
            "range": "stddev: 0.00014414907183470666",
            "extra": "mean: 35.879001314579334 msec\nrounds: 27"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[128]",
            "value": 10.873903017315794,
            "unit": "iter/sec",
            "range": "stddev: 0.00025974410719497356",
            "extra": "mean: 91.96329950778322 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[1]",
            "value": 81.45871118290421,
            "unit": "iter/sec",
            "range": "stddev: 0.00016714386092238795",
            "extra": "mean: 12.276157890033872 msec\nrounds: 79"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[128]",
            "value": 23.27925610426345,
            "unit": "iter/sec",
            "range": "stddev: 0.00025667394235683684",
            "extra": "mean: 42.95669911105348 msec\nrounds: 23"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[1]",
            "value": 42.123801274438634,
            "unit": "iter/sec",
            "range": "stddev: 0.00008524100283330402",
            "extra": "mean: 23.739547945470328 msec\nrounds: 42"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[128]",
            "value": 41.2288793302751,
            "unit": "iter/sec",
            "range": "stddev: 0.00007870420209114794",
            "extra": "mean: 24.254843115895277 msec\nrounds: 42"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[1]",
            "value": 54.859515933632466,
            "unit": "iter/sec",
            "range": "stddev: 0.00006658297583143198",
            "extra": "mean: 18.22837812148712 msec\nrounds: 55"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[128]",
            "value": 54.08178253574209,
            "unit": "iter/sec",
            "range": "stddev: 0.0002435711303675045",
            "extra": "mean: 18.490514792834542 msec\nrounds: 56"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[1]",
            "value": 32.40426636235144,
            "unit": "iter/sec",
            "range": "stddev: 0.00018334897915059966",
            "extra": "mean: 30.860133934766054 msec\nrounds: 33"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[128]",
            "value": 32.83268704954255,
            "unit": "iter/sec",
            "range": "stddev: 0.0005984593423362888",
            "extra": "mean: 30.45745230937267 msec\nrounds: 34"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[1]",
            "value": 30.207838403379107,
            "unit": "iter/sec",
            "range": "stddev: 0.00031629957619277923",
            "extra": "mean: 33.10399064794183 msec\nrounds: 30"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[128]",
            "value": 13.989751957989697,
            "unit": "iter/sec",
            "range": "stddev: 0.00048116675761147",
            "extra": "mean: 71.48089565868888 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[1]",
            "value": 6.236663628121926,
            "unit": "iter/sec",
            "range": "stddev: 0.0019929733084377045",
            "extra": "mean: 160.34214118761676 msec\nrounds: 7"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[128]",
            "value": 0.8410226591925327,
            "unit": "iter/sec",
            "range": "stddev: 0.0043870501464433834",
            "extra": "mean: 1.1890286059118806 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[1]",
            "value": 5.749120763445503,
            "unit": "iter/sec",
            "range": "stddev: 0.0005852846684172309",
            "extra": "mean: 173.9396407113721 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[128]",
            "value": 3.3140394582625747,
            "unit": "iter/sec",
            "range": "stddev: 0.004600214081891292",
            "extra": "mean: 301.7465581186116 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[1]",
            "value": 4.82548936159077,
            "unit": "iter/sec",
            "range": "stddev: 0.0006323667020391208",
            "extra": "mean: 207.23286801949143 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[128]",
            "value": 2.637068818193202,
            "unit": "iter/sec",
            "range": "stddev: 0.0022218667667523967",
            "extra": "mean: 379.20891297981143 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Filippo Luca Ferretti",
            "username": "flferretti",
            "email": "102977828+flferretti@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "1e5b87c8c58b5391e67c38fe99a1af9e8835fa49",
          "message": "Merge pull request #441 from ami-iit/add_lfs_instructions\n\nAdd LFS instructions for Pixi installation",
          "timestamp": "2025-06-05T10:10:23Z",
          "url": "https://github.com/ami-iit/jaxsim/commit/1e5b87c8c58b5391e67c38fe99a1af9e8835fa49"
        },
        "date": 1749428467589,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[1]",
            "value": 33.82466444520671,
            "unit": "iter/sec",
            "range": "stddev: 0.00012511924274217413",
            "extra": "mean: 29.564225289504975 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[128]",
            "value": 15.737312357474245,
            "unit": "iter/sec",
            "range": "stddev: 0.0002084721368138263",
            "extra": "mean: 63.54325168649666 msec\nrounds: 16"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[1]",
            "value": 27.191497889084015,
            "unit": "iter/sec",
            "range": "stddev: 0.0002607893051159415",
            "extra": "mean: 36.77620129935719 msec\nrounds: 27"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[128]",
            "value": 11.292671283266202,
            "unit": "iter/sec",
            "range": "stddev: 0.0006135397464032982",
            "extra": "mean: 88.5530070712169 msec\nrounds: 12"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[1]",
            "value": 78.6530895676359,
            "unit": "iter/sec",
            "range": "stddev: 0.00016337857724122175",
            "extra": "mean: 12.714058729251484 msec\nrounds: 77"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[128]",
            "value": 23.86774695684349,
            "unit": "iter/sec",
            "range": "stddev: 0.00036592519556304126",
            "extra": "mean: 41.897544908958174 msec\nrounds: 24"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[1]",
            "value": 40.455836855777214,
            "unit": "iter/sec",
            "range": "stddev: 0.0003204484395729559",
            "extra": "mean: 24.7183120587752 msec\nrounds: 42"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[128]",
            "value": 40.268236542405475,
            "unit": "iter/sec",
            "range": "stddev: 0.0003349736369569946",
            "extra": "mean: 24.83346890413055 msec\nrounds: 38"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[1]",
            "value": 52.643239514429034,
            "unit": "iter/sec",
            "range": "stddev: 0.00021870504616368014",
            "extra": "mean: 18.99579146769471 msec\nrounds: 52"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[128]",
            "value": 53.173172563862344,
            "unit": "iter/sec",
            "range": "stddev: 0.0000811905011510118",
            "extra": "mean: 18.80647611911767 msec\nrounds: 52"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[1]",
            "value": 31.452990109900398,
            "unit": "iter/sec",
            "range": "stddev: 0.0006162626337388626",
            "extra": "mean: 31.793479618499987 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[128]",
            "value": 31.929681983546324,
            "unit": "iter/sec",
            "range": "stddev: 0.0003537486968847584",
            "extra": "mean: 31.318821168194212 msec\nrounds: 33"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[1]",
            "value": 29.496111213080216,
            "unit": "iter/sec",
            "range": "stddev: 0.00006999611563651067",
            "extra": "mean: 33.90277425983343 msec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[128]",
            "value": 14.219973487495901,
            "unit": "iter/sec",
            "range": "stddev: 0.0003243081329970453",
            "extra": "mean: 70.32361916000289 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[1]",
            "value": 6.250947319307351,
            "unit": "iter/sec",
            "range": "stddev: 0.0006710272957046258",
            "extra": "mean: 159.97575230098195 msec\nrounds: 7"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[128]",
            "value": 0.8456352035674847,
            "unit": "iter/sec",
            "range": "stddev: 0.0006822541329527538",
            "extra": "mean: 1.1825430111959576 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[1]",
            "value": 5.718060611193069,
            "unit": "iter/sec",
            "range": "stddev: 0.002810183722077737",
            "extra": "mean: 174.88447010206679 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[128]",
            "value": 3.372294112118496,
            "unit": "iter/sec",
            "range": "stddev: 0.0006073453264689539",
            "extra": "mean: 296.53404084965587 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[1]",
            "value": 4.6047703767890935,
            "unit": "iter/sec",
            "range": "stddev: 0.0004822255381336423",
            "extra": "mean: 217.1660947613418 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[128]",
            "value": 2.6316031627521377,
            "unit": "iter/sec",
            "range": "stddev: 0.0014905312672725199",
            "extra": "mean: 379.99650333076715 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Carlotta Sartore",
            "username": "CarlottaSartore",
            "email": "56030908+CarlottaSartore@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "25109f59cc85551bec9328854f5e1571555489b5",
          "message": "Merge pull request #442 from ami-iit/CarlottaSartore-patch-1\n\nUpdate readme structure and video content",
          "timestamp": "2025-06-09T14:03:45Z",
          "url": "https://github.com/ami-iit/jaxsim/commit/25109f59cc85551bec9328854f5e1571555489b5"
        },
        "date": 1750033252106,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[1]",
            "value": 33.9247775322331,
            "unit": "iter/sec",
            "range": "stddev: 0.000145486698925284",
            "extra": "mean: 29.476980329491198 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[128]",
            "value": 15.760947735131491,
            "unit": "iter/sec",
            "range": "stddev: 0.00039466552087795306",
            "extra": "mean: 63.44796117627993 msec\nrounds: 16"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[1]",
            "value": 27.117361340238325,
            "unit": "iter/sec",
            "range": "stddev: 0.0003073682156212765",
            "extra": "mean: 36.876744291345986 msec\nrounds: 28"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[128]",
            "value": 11.373566810931507,
            "unit": "iter/sec",
            "range": "stddev: 0.0002842441181029588",
            "extra": "mean: 87.92316576000303 msec\nrounds: 12"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[1]",
            "value": 78.60017943294385,
            "unit": "iter/sec",
            "range": "stddev: 0.00019074158250704729",
            "extra": "mean: 12.722617266454586 msec\nrounds: 76"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[128]",
            "value": 24.47499314186665,
            "unit": "iter/sec",
            "range": "stddev: 0.0003296931249763229",
            "extra": "mean: 40.85802983492613 msec\nrounds: 25"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[1]",
            "value": 40.9942257427606,
            "unit": "iter/sec",
            "range": "stddev: 0.0001404281942017374",
            "extra": "mean: 24.393679399508983 msec\nrounds: 40"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[128]",
            "value": 40.44841345234631,
            "unit": "iter/sec",
            "range": "stddev: 0.00007560194071967156",
            "extra": "mean: 24.722848553210497 msec\nrounds: 40"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[1]",
            "value": 53.26596131393606,
            "unit": "iter/sec",
            "range": "stddev: 0.00008848529316461428",
            "extra": "mean: 18.77371543350647 msec\nrounds: 53"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[128]",
            "value": 53.40567691312945,
            "unit": "iter/sec",
            "range": "stddev: 0.00014203950673913923",
            "extra": "mean: 18.72460116228124 msec\nrounds: 53"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[1]",
            "value": 31.061853794897566,
            "unit": "iter/sec",
            "range": "stddev: 0.0015487195803080033",
            "extra": "mean: 32.19382869428955 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[128]",
            "value": 32.02660949060554,
            "unit": "iter/sec",
            "range": "stddev: 0.0003283454462850203",
            "extra": "mean: 31.2240357598057 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[1]",
            "value": 29.878446483395575,
            "unit": "iter/sec",
            "range": "stddev: 0.00013155995891146605",
            "extra": "mean: 33.468942254267894 msec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[128]",
            "value": 14.52324428456363,
            "unit": "iter/sec",
            "range": "stddev: 0.0013236438293721365",
            "extra": "mean: 68.8551387283951 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[1]",
            "value": 6.135920398192636,
            "unit": "iter/sec",
            "range": "stddev: 0.0005175622805684227",
            "extra": "mean: 162.97473485714622 msec\nrounds: 7"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[128]",
            "value": 0.8452854242919515,
            "unit": "iter/sec",
            "range": "stddev: 0.0004952203032909414",
            "extra": "mean: 1.183032347727567 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[1]",
            "value": 5.637930365716057,
            "unit": "iter/sec",
            "range": "stddev: 0.0023860292391008177",
            "extra": "mean: 177.3700516205281 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[128]",
            "value": 3.3820092813479565,
            "unit": "iter/sec",
            "range": "stddev: 0.0005500156435498118",
            "extra": "mean: 295.6822163425386 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[1]",
            "value": 4.690963523109687,
            "unit": "iter/sec",
            "range": "stddev: 0.0014683208439723585",
            "extra": "mean: 213.17582093179226 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[128]",
            "value": 2.6479304828968346,
            "unit": "iter/sec",
            "range": "stddev: 0.001180183016529729",
            "extra": "mean: 377.65341894701123 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Carlotta Sartore",
            "username": "CarlottaSartore",
            "email": "56030908+CarlottaSartore@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "25109f59cc85551bec9328854f5e1571555489b5",
          "message": "Merge pull request #442 from ami-iit/CarlottaSartore-patch-1\n\nUpdate readme structure and video content",
          "timestamp": "2025-06-09T14:03:45Z",
          "url": "https://github.com/ami-iit/jaxsim/commit/25109f59cc85551bec9328854f5e1571555489b5"
        },
        "date": 1750638091350,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[1]",
            "value": 33.73961764099612,
            "unit": "iter/sec",
            "range": "stddev: 0.0002799885046157775",
            "extra": "mean: 29.638747262652032 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[128]",
            "value": 15.726891986288352,
            "unit": "iter/sec",
            "range": "stddev: 0.0004903548820907823",
            "extra": "mean: 63.5853543644771 msec\nrounds: 16"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[1]",
            "value": 27.10321559864387,
            "unit": "iter/sec",
            "range": "stddev: 0.00021106968635234698",
            "extra": "mean: 36.89599104432596 msec\nrounds: 27"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[128]",
            "value": 11.449475418584834,
            "unit": "iter/sec",
            "range": "stddev: 0.0005140710349729022",
            "extra": "mean: 87.34024603230257 msec\nrounds: 12"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[1]",
            "value": 78.49204319014721,
            "unit": "iter/sec",
            "range": "stddev: 0.0001388705734755183",
            "extra": "mean: 12.740144852357798 msec\nrounds: 78"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[128]",
            "value": 24.2525614347928,
            "unit": "iter/sec",
            "range": "stddev: 0.0002414530193829016",
            "extra": "mean: 41.232758143451065 msec\nrounds: 24"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[1]",
            "value": 40.759662595751735,
            "unit": "iter/sec",
            "range": "stddev: 0.00023901698960146007",
            "extra": "mean: 24.53405981099135 msec\nrounds: 38"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[128]",
            "value": 40.4207752299656,
            "unit": "iter/sec",
            "range": "stddev: 0.00023593503168485306",
            "extra": "mean: 24.739753117319196 msec\nrounds: 40"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[1]",
            "value": 52.89530298002137,
            "unit": "iter/sec",
            "range": "stddev: 0.00023086772538318125",
            "extra": "mean: 18.905270291725174 msec\nrounds: 48"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[128]",
            "value": 53.08595520838406,
            "unit": "iter/sec",
            "range": "stddev: 0.00022109623836497083",
            "extra": "mean: 18.837374143021286 msec\nrounds: 53"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[1]",
            "value": 31.40065154389751,
            "unit": "iter/sec",
            "range": "stddev: 0.0003104261961810635",
            "extra": "mean: 31.846472949837334 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[128]",
            "value": 31.975792325908156,
            "unit": "iter/sec",
            "range": "stddev: 0.0009126622383582198",
            "extra": "mean: 31.273658203920633 msec\nrounds: 33"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[1]",
            "value": 29.58260317621871,
            "unit": "iter/sec",
            "range": "stddev: 0.00025448848917166865",
            "extra": "mean: 33.803651221738804 msec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[128]",
            "value": 14.65077627316994,
            "unit": "iter/sec",
            "range": "stddev: 0.0012751251160196177",
            "extra": "mean: 68.25576893364389 msec\nrounds: 15"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[1]",
            "value": 6.160606364112983,
            "unit": "iter/sec",
            "range": "stddev: 0.0002908668350073156",
            "extra": "mean: 162.3216840837683 msec\nrounds: 7"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[128]",
            "value": 0.8437915381747435,
            "unit": "iter/sec",
            "range": "stddev: 0.00188760247120757",
            "extra": "mean: 1.1851268408820033 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[1]",
            "value": 5.752731567649072,
            "unit": "iter/sec",
            "range": "stddev: 0.0011134089156764446",
            "extra": "mean: 173.8304644046972 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[128]",
            "value": 3.371693485571547,
            "unit": "iter/sec",
            "range": "stddev: 0.0027302750116520494",
            "extra": "mean: 296.58686481416225 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[1]",
            "value": 4.682505302546448,
            "unit": "iter/sec",
            "range": "stddev: 0.0004250566071764865",
            "extra": "mean: 213.56089003384113 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[128]",
            "value": 2.662639645338311,
            "unit": "iter/sec",
            "range": "stddev: 0.0008640506440978913",
            "extra": "mean: 375.56715635582805 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Filippo Luca Ferretti",
            "username": "flferretti",
            "email": "102977828+flferretti@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "eaf82a71556e49a08a454bb19bc17d960fb6248d",
          "message": "Merge pull request #461 from ami-iit/flferretti-patch-1\n\nUpdate bots name to be excluded from releases",
          "timestamp": "2025-09-09T12:29:59Z",
          "url": "https://github.com/ami-iit/jaxsim/commit/eaf82a71556e49a08a454bb19bc17d960fb6248d"
        },
        "date": 1757427771926,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[1]",
            "value": 392.83994442296586,
            "unit": "iter/sec",
            "range": "stddev: 0.000008844683095886968",
            "extra": "mean: 2.5455659848157204 msec\nrounds: 395"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_dynamics_aba[128]",
            "value": 29.24995678393885,
            "unit": "iter/sec",
            "range": "stddev: 0.00010557873673903868",
            "extra": "mean: 34.18808469997809 msec\nrounds: 30"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[1]",
            "value": 362.5130449271654,
            "unit": "iter/sec",
            "range": "stddev: 0.000008641731407417134",
            "extra": "mean: 2.7585214214868206 msec\nrounds: 363"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_bias_forces[128]",
            "value": 16.27499626055415,
            "unit": "iter/sec",
            "range": "stddev: 0.00019610825022008387",
            "extra": "mean: 61.443946529420025 msec\nrounds: 17"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[1]",
            "value": 455.3542246444632,
            "unit": "iter/sec",
            "range": "stddev: 0.000006943575555738578",
            "extra": "mean: 2.1960925053035174 msec\nrounds: 849"
          },
          {
            "name": "tests/test_benchmark.py::test_forward_kinematics[128]",
            "value": 31.918573112322537,
            "unit": "iter/sec",
            "range": "stddev: 0.00017857766288173826",
            "extra": "mean: 31.32972130304717 msec\nrounds: 33"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[1]",
            "value": 184.85633166611,
            "unit": "iter/sec",
            "range": "stddev: 0.00003945770719180179",
            "extra": "mean: 5.4096064277972005 msec\nrounds: 187"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_mass_matrix[128]",
            "value": 176.03804507334434,
            "unit": "iter/sec",
            "range": "stddev: 0.00001728009903516206",
            "extra": "mean: 5.680590235953603 msec\nrounds: 178"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[1]",
            "value": 533.9995336171206,
            "unit": "iter/sec",
            "range": "stddev: 0.000006755488190166284",
            "extra": "mean: 1.8726608115673062 msec\nrounds: 536"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian[128]",
            "value": 539.4262993450989,
            "unit": "iter/sec",
            "range": "stddev: 0.000006925503885030166",
            "extra": "mean: 1.8538213676531339 msec\nrounds: 544"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[1]",
            "value": 416.8168050954244,
            "unit": "iter/sec",
            "range": "stddev: 0.00001490457152244306",
            "extra": "mean: 2.3991355141524675 msec\nrounds: 424"
          },
          {
            "name": "tests/test_benchmark.py::test_free_floating_jacobian_derivative[128]",
            "value": 302.87335657365946,
            "unit": "iter/sec",
            "range": "stddev: 0.000015935779304446555",
            "extra": "mean: 3.3017100325785766 msec\nrounds: 307"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[1]",
            "value": 351.25588853970714,
            "unit": "iter/sec",
            "range": "stddev: 0.000009962003016838646",
            "extra": "mean: 2.8469273615806068 msec\nrounds: 354"
          },
          {
            "name": "tests/test_benchmark.py::test_soft_contact_model[128]",
            "value": 29.868841897404145,
            "unit": "iter/sec",
            "range": "stddev: 0.00015866943564363386",
            "extra": "mean: 33.4797044838524 msec\nrounds: 31"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[1]",
            "value": 40.37782360742885,
            "unit": "iter/sec",
            "range": "stddev: 0.00009135791373874213",
            "extra": "mean: 24.76606985365146 msec\nrounds: 41"
          },
          {
            "name": "tests/test_benchmark.py::test_rigid_contact_model[128]",
            "value": 0.7260623838134609,
            "unit": "iter/sec",
            "range": "stddev: 0.0003105221536316684",
            "extra": "mean: 1.3772921202001271 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[1]",
            "value": 75.07584606499664,
            "unit": "iter/sec",
            "range": "stddev: 0.00003332995094618044",
            "extra": "mean: 13.319863210522511 msec\nrounds: 76"
          },
          {
            "name": "tests/test_benchmark.py::test_relaxed_rigid_contact_model[128]",
            "value": 6.4181486112957105,
            "unit": "iter/sec",
            "range": "stddev: 0.00016129290730799092",
            "extra": "mean: 155.80817157147717 msec\nrounds: 7"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[1]",
            "value": 69.48281363997596,
            "unit": "iter/sec",
            "range": "stddev: 0.00003366769986653002",
            "extra": "mean: 14.392048157138301 msec\nrounds: 70"
          },
          {
            "name": "tests/test_benchmark.py::test_simulation_step[128]",
            "value": 5.390796419952395,
            "unit": "iter/sec",
            "range": "stddev: 0.0003737239121214081",
            "extra": "mean: 185.50134750012148 msec\nrounds: 6"
          }
        ]
      }
    ]
  }
}