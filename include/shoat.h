#ifndef SHOAT_H
#define SHOAT_H
#include <cassert>
#include <climits>

class shoat
{
    public:
        shoat(float);
        shoat(unsigned short);
        virtual ~shoat();
    protected:
    private:
        unsigned short value;
};

#endif // SHOAT_H
