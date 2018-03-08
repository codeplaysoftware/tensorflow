#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_SYCL_FAST_DIV_H_
#define TENSORFLOW_KERNELS_CONV_OPS_SYCL_FAST_DIV_H_

namespace tensorflow {
namespace fast_div {
/** This uses the fast integer division technique outlined in "Division by
 * Invariant Integers using Multiplication" by Granlund and Montgomery
 * (http://dx.doi.org/10.1145/773473.178249), and the implementation is based
 * on that found in Chapter 10 (Figure 10-1) of "Hackers Delight" by Warren.
 *
 * The idea behind this fast division algorithm is to perform some additional
 * computations on the host to compute suitable magic numbers to convert each
 * division on the device into a multiply followed by a shift.
 *
 * The key component to this is the mul_hi operation, which takes two integers
 * and multiplies them using twice the number of bits before returning the top
 * half of the bits. In the 32 bit case, this is equivalent to performing a 64
 * bit multiply and shifting the result left by 32. Mathematically this is
 * equivalent to:
 *
 *   mul_hi(x, y) = floor(x * y / 2^32)
 *
 * If the mul_hi operation is followed by a shift left by 'z' bits, then the
 * whole fast division is equivalent to:
 *
 *     fast_div(x, y, z) = mul_hi(x, y) >> z = floor(mul_hi(x, y) / 2^z) =
 *     floor( floor(x * y / 2^32) / 2^z) = floor( x * y / 2^(32 + z) )
 *
 * More generally, for W-bit integers, for a given divisor 'd', we need the
 * smallest multiple 'm' and shift 's' satisfying:
 *
 *     floor(m * n / 2^(W + s)) = floor(n / d)
 *
 * for every possible signed integer 'n' where 0 <= n < 2^(W-1).
 *
 * The smallest such multiple can be any integer between 0 and 2^W, however the
 * largest representable integer in the signed integer is 2^(W-1), so the
 * multiple must be stored in an unsigned integer and the mul_hi operation must
 * also be computed using unsigned types.
 *
 * Let 'p = W + s', then we need 'm' to be the next integer greater than '2^p /
 * d', that is
 *
 * (1)  m = (2^p + d - (2^p % d) ) / d
 *
 * We can find 'p' by using the largest representable integer 'nc' such that
 * (nc % d) = d - 1, or equivalently
 *
 *     nc = 2^(W-1) - (2^(W-1) % d) - 1
 *
 * Then p can be found using the inequality:
 *
 * (2)  2^p > nc * ( d - (2^p, d) )
 *
 * and the fact that if 'p_0' satisfies this, then so does 'p_0 + 1'.
 *
 * We know 'p' is at least W, so starting with this we can try each value of
 * 'p' until we find the smallest value satsifying (2). This will give the
 * shift value 's = p - W', and (1) will give the value for m.
 *
 * In this implementation we assume that the divisor is positive, which allows
 * us to skip certain branches and checks otherwise required. This approach
 * also only works for divisors strictly greater than 1.
*/
template <typename index_type>
struct magic_numbers {
  static_assert(std::is_signed<index_type>::value,
                "Index type for fast division must be a signed type.");
  using unsigned_type = typename std::make_unsigned<index_type>::type;
  magic_numbers(index_type divisor) {
    assert(divisor > 1);

    constexpr int index_bit_length = std::numeric_limits<index_type>::digits;
    constexpr unsigned_type two_pow = static_cast<unsigned_type>(1)
                                      << index_bit_length;

    const unsigned_type unsigned_d = static_cast<unsigned_type>(divisor);
    const unsigned_type nc = two_pow - 1 - (two_pow % unsigned_d);

    int power = index_bit_length;
    unsigned_type two_p_quot_nc = two_pow / nc;
    unsigned_type two_p_rem_nc = two_pow % nc;
    unsigned_type two_p_quot_d = two_pow / unsigned_d;
    unsigned_type two_p_rem_d = two_pow % unsigned_d;

    auto increase_two_power_by_one = [](unsigned_type div, unsigned_type& quot,
                                        unsigned_type& rem) {
      quot *= 2;
      rem *= 2;
      if (rem >= div) {
        ++quot;
        rem -= div;
      }
    };

    unsigned_type delta;
    do {
      ++power;
      increase_two_power_by_one(nc, two_p_quot_nc, two_p_rem_nc);
      increase_two_power_by_one(unsigned_d, two_p_quot_d, two_p_rem_d);

      delta = unsigned_d - two_p_rem_d;

    } while (two_p_quot_nc < delta ||
             (two_p_quot_nc == delta && two_p_rem_nc == 0));

    multiple = two_p_quot_d + 1;
    shift = power - index_bit_length - 1;
  }

  unsigned_type multiple;
  index_type shift;
};
template <typename index_type>
index_type inline TF_ATTRIBUTE_ALWAYS_INLINE divide(
    index_type value, typename std::make_unsigned<index_type>::type multiple,
    index_type shift) {
  using unsigned_type = typename std::make_unsigned<index_type>::type;
  assert(value > 0);
  unsigned_type unsigned_value = static_cast<unsigned_type>(value);
  unsigned_type unsigned_ans =
      cl::sycl::mul_hi(unsigned_value, multiple) >> shift;
  return static_cast<index_type>(unsigned_ans);
}
template <typename index_type>
index_type inline TF_ATTRIBUTE_ALWAYS_INLINE divide(
    index_type value, magic_numbers<index_type> magic) {
  return divide(value, magic.multiple, magic.shift);
}
template <typename index_type>
index_type inline TF_ATTRIBUTE_ALWAYS_INLINE operator/(
    index_type value, magic_numbers<index_type> magic) {
  static_assert(std::is_signed<index_type>::value,
                "Fast division is only supported on signed integer types.");
  return divide(value, magic.multiple, magic.shift);
}
template <typename Index, bool use_fast_div>
struct index_div {
  using type = Index;
};
template <typename Index>
struct index_div<Index, true> {
  using type = magic_numbers<Index>;
};
}  // namespace fast_div
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_CONV_OPS_SYCL_FAST_DIV_H_
