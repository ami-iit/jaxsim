window.BENCHMARK_DATA = {
  "lastUpdate": 1744590059441,
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
      }
    ]
  }
}