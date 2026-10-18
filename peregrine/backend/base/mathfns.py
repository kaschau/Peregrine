import math
import struct
from decimal import Decimal

LN2 = 0.6931471805599453

# log x = e ln 2 + log m_k + log(1 + u) for x = m 2^e, m_k the centre of m's bin by its top LOGBITS bits, u = m / m_k - 1
LOGBITS = 7
LOGBINS = 1 << LOGBITS
LOGCENTRES = [0.5 + (k + 0.5) / (2 * LOGBINS) for k in range(LOGBINS)]
LOGTAB = [math.log(m) for m in LOGCENTRES]
RCPTAB = [1.0 / m for m in LOGCENTRES]
# exp x = 2^n 2^(j / EXPBINS) e^r for k = n EXPBINS + j the integer nearest x EXPBINS / ln 2, |r| below ln 2 / (2 EXPBINS)
EXPBITS = 6
EXPBINS = 1 << EXPBITS
EXPTAB = [2.0 ** (j / EXPBINS) for j in range(EXPBINS)]
_LN2 = Decimal("0.6931471805599453094172321214581765680755")
EXPK = float(EXPBINS / _LN2)
# ln 2 / EXPBINS in two parts, the head's low 20 bits clear so k times it is exact
_head = struct.unpack("<q", struct.pack("<d", float(_LN2 / EXPBINS)))[0] & ~((1 << 20) - 1)
LN2BINS_HI = struct.unpack("<d", struct.pack("<q", _head))[0]
LN2BINS_LO = float(_LN2 / EXPBINS - Decimal(LN2BINS_HI))
