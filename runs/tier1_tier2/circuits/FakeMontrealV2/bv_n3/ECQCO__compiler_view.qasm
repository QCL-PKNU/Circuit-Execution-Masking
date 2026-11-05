OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[3];
x q[0];
barrier q[0],q[4],q[1];
measure q[0] -> meas[0];
measure q[4] -> meas[1];
measure q[1] -> meas[2];