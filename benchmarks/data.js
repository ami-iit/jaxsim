window.BENCHMARK_DATA = {
  "lastUpdate": 1741691125333,
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
      }
    ]
  }
}