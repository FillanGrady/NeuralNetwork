#include "shoat.h"

shoat::shoat(float f)
{
    assert(0 <= f && f <= 1);
    value = unsigned short(f * USHRT_MAX);
}

shoat::shoat(unsigned short u)
{
    value = unsigned short(u);
}

shoat shoat::operator+(const& shoat b)
{
    return this->value + b->value;
}

shoat shoat::operator-(const& shoat b)
{
    return this->value - b->value;
}

shoat::~shoat()
{
    //dtor
}
