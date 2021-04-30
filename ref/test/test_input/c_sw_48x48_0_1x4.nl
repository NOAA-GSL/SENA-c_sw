&io
  input_data_dir='test/data/inputs',
  input_file='c_sw_48x48.nc',
  output_data_dir='data/outputs',
  output_file='c_sw_48x48_0.nc'
  log_dir='test_output'
  log_file='c_sw_48x48_0.log'
  interpFactor = 0
/

&debug
  do_profile=0
/

&decomposition
  rows=1
  cols=4
/
