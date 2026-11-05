OPENQASM 2.0;
include "qelib1.inc";
qreg q[116];
creg meas[3];
x q[24];
barrier q[24],q[48],q[26];
measure q[24] -> meas[0];
measure q[48] -> meas[1];
measure q[26] -> meas[2];