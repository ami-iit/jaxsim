window.BENCHMARK_DATA = {
  "lastUpdate": 1742170758484,
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
      }
    ]
  }
}