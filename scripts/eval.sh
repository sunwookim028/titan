make test EXE="titan.print" OPT="-g 1 -l -b" HG=76bp SIZE=A ERRBUF=76bp.err OUTBUF=76bp.A.out.base
make test EXE="titan.print" OPT="-g 1 -l -b" HG=76bp SIZE=B ERRBUF=76bp.err OUTBUF=76bp.B.out.base
make test EXE="titan.print" OPT="-g 1 -l -b" HG=152bp SIZE=A ERRBUF=152bp.err OUTBUF=152bp.A.out.base
make test EXE="titan.print" OPT="-g 1 -l -b" HG=152bp SIZE=B ERRBUF=152bp.err OUTBUF=152bp.B.out.base
make test EXE="titan.print" OPT="-g 1 -l -b" HG=251bp SIZE=A ERRBUF=251bp.err OUTBUF=251bp.A.out.base
make test EXE="titan.print" OPT="-g 1 -l -b" HG=251bp SIZE=B ERRBUF=251bp.err OUTBUF=251bp.B.out.base

make test EXE="titan.print" OPT="-g 1 -l " HG=76bp SIZE=A ERRBUF=76bp.err OUTBUF=76bp.A.out.titan
make test EXE="titan.print" OPT="-g 1 -l " HG=76bp SIZE=B ERRBUF=76bp.err OUTBUF=76bp.B.out.titan
make test EXE="titan.print" OPT="-g 1 -l " HG=152bp SIZE=A ERRBUF=152bp.err OUTBUF=152bp.A.out.titan
make test EXE="titan.print" OPT="-g 1 -l " HG=152bp SIZE=B ERRBUF=152bp.err OUTBUF=152bp.B.out.titan
make test EXE="titan.print" OPT="-g 1 -l " HG=251bp SIZE=A ERRBUF=251bp.err OUTBUF=251bp.A.out.titan
make test EXE="titan.print" OPT="-g 1 -l " HG=251bp SIZE=B ERRBUF=251bp.err OUTBUF=251bp.B.out.titan

/usr/bin/grep SMintv 76bp.A.out.titan | sed 's/\[//' | sed 's/\]//' | sort -s -n -k 4 | sort -s -n -k 3 | sort -s -n -k 2 > 76bp.SMintv.titan
/usr/bin/grep SMintv 76bp.B.out.titan | sed 's/\[//' | sed 's/\]//' | sort -s -n -k 4 | sort -s -n -k 3 | sort -s -n -k 2 | awk '{ $2 = $2 + 50000; print }' >> 76bp.SMintv.titan
/usr/bin/grep SMintv 152bp.A.out.titan | sed 's/\[//' | sed 's/\]//' | sort -s -n -k 4 | sort -s -n -k 3 | sort -s -n -k 2 > 152bp.SMintv.titan
/usr/bin/grep SMintv 152bp.B.out.titan | sed 's/\[//' | sed 's/\]//' | sort -s -n -k 4 | sort -s -n -k 3 | sort -s -n -k 2 | awk '{ $2 = $2 + 50000; print }' >> 152bp.SMintv.titan
/usr/bin/grep SMintv 251bp.A.out.titan | sed 's/\[//' | sed 's/\]//' | sort -s -n -k 4 | sort -s -n -k 3 | sort -s -n -k 2 > 251bp.SMintv.titan
/usr/bin/grep SMintv 251bp.B.out.titan | sed 's/\[//' | sed 's/\]//' | sort -s -n -k 4 | sort -s -n -k 3 | sort -s -n -k 2 | awk '{ $2 = $2 + 50000; print }' >> 251bp.SMintv.titan

/usr/bin/grep SMintv 76bp.A.out.base | sed 's/\[//' | sed 's/\]//' | sort -s -n -k 4 | sort -s -n -k 3 | sort -s -n -k 2 > 76bp.SMintv.base
/usr/bin/grep SMintv 76bp.B.out.base | sed 's/\[//' | sed 's/\]//' | sort -s -n -k 4 | sort -s -n -k 3 | sort -s -n -k 2 | awk '{ $2 = $2 + 50000; print }' >> 76bp.SMintv.base
/usr/bin/grep SMintv 152bp.A.out.base | sed 's/\[//' | sed 's/\]//' | sort -s -n -k 4 | sort -s -n -k 3 | sort -s -n -k 2 > 152bp.SMintv.base
/usr/bin/grep SMintv 152bp.B.out.base | sed 's/\[//' | sed 's/\]//' | sort -s -n -k 4 | sort -s -n -k 3 | sort -s -n -k 2 | awk '{ $2 = $2 + 50000; print }' >> 152bp.SMintv.base
/usr/bin/grep SMintv 251bp.A.out.base | sed 's/\[//' | sed 's/\]//' | sort -s -n -k 4 | sort -s -n -k 3 | sort -s -n -k 2 > 251bp.SMintv.base
/usr/bin/grep SMintv 251bp.B.out.base | sed 's/\[//' | sed 's/\]//' | sort -s -n -k 4 | sort -s -n -k 3 | sort -s -n -k 2 | awk '{ $2 = $2 + 50000; print }' >> 251bp.SMintv.base

/usr/bin/grep CHintv 76bp.A.out.titan | sed 's/\[//' | sed 's/\]//' | sort -s -n -k 4 | sort -s -n -k 3 | sort -s -n -k 2 > 76bp.CHintv.titan
/usr/bin/grep CHintv 76bp.B.out.titan | sed 's/\[//' | sed 's/\]//' | sort -s -n -k 4 | sort -s -n -k 3 | sort -s -n -k 2 | awk '{ $2 = $2 + 50000; print }' >> 76bp.CHintv.titan
/usr/bin/grep CHintv 152bp.A.out.titan | sed 's/\[//' | sed 's/\]//' | sort -s -n -k 4 | sort -s -n -k 3 | sort -s -n -k 2 > 152bp.CHintv.titan
/usr/bin/grep CHintv 152bp.B.out.titan | sed 's/\[//' | sed 's/\]//' | sort -s -n -k 4 | sort -s -n -k 3 | sort -s -n -k 2 | awk '{ $2 = $2 + 50000; print }' >> 152bp.CHintv.titan
/usr/bin/grep CHintv 251bp.A.out.titan | sed 's/\[//' | sed 's/\]//' | sort -s -n -k 4 | sort -s -n -k 3 | sort -s -n -k 2 > 251bp.CHintv.titan
/usr/bin/grep CHintv 251bp.B.out.titan | sed 's/\[//' | sed 's/\]//' | sort -s -n -k 4 | sort -s -n -k 3 | sort -s -n -k 2 | awk '{ $2 = $2 + 50000; print }' >> 251bp.CHintv.titan

/usr/bin/grep CHintv 76bp.A.out.base | sed 's/\[//' | sed 's/\]//' | sort -s -n -k 4 | sort -s -n -k 3 | sort -s -n -k 2 > 76bp.CHintv.base
/usr/bin/grep CHintv 76bp.B.out.base | sed 's/\[//' | sed 's/\]//' | sort -s -n -k 4 | sort -s -n -k 3 | sort -s -n -k 2 | awk '{ $2 = $2 + 50000; print }' >> 76bp.CHintv.base
/usr/bin/grep CHintv 152bp.A.out.base | sed 's/\[//' | sed 's/\]//' | sort -s -n -k 4 | sort -s -n -k 3 | sort -s -n -k 2 > 152bp.CHintv.base
/usr/bin/grep CHintv 152bp.B.out.base | sed 's/\[//' | sed 's/\]//' | sort -s -n -k 4 | sort -s -n -k 3 | sort -s -n -k 2 | awk '{ $2 = $2 + 50000; print }' >> 152bp.CHintv.base
/usr/bin/grep CHintv 251bp.A.out.base | sed 's/\[//' | sed 's/\]//' | sort -s -n -k 4 | sort -s -n -k 3 | sort -s -n -k 2 > 251bp.CHintv.base
/usr/bin/grep CHintv 251bp.B.out.base | sed 's/\[//' | sed 's/\]//' | sort -s -n -k 4 | sort -s -n -k 3 | sort -s -n -k 2 | awk '{ $2 = $2 + 50000; print }' >> 251bp.CHintv.base
