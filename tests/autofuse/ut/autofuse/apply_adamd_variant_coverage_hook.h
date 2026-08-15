/*
 * Test-only coverage hook for the three-variant std::visit in ApplyAdamD
 * lowering. This header is force-included only when building the autofusion
 * unit-test target and is never part of the production build.
 */
#ifndef AUTOFUSE_APPLY_ADAMD_VARIANT_COVERAGE_HOOK_H_
#define AUTOFUSE_APPLY_ADAMD_VARIANT_COVERAGE_HOOK_H_

#include <cstddef>
#include <type_traits>
#include <utility>
#include <variant>

namespace autofuse_variant_coverage_detail {
template <typename T>
using RemoveCvRef = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename Visitor, typename Variant1, typename Variant2, typename Variant3, std::size_t I1, std::size_t I2,
          std::size_t I3>
void InvokeOne(Visitor &visitor);

template <typename Visitor, typename Variant1, typename Variant2, typename Variant3, std::size_t I1, std::size_t I2,
          std::size_t... I3>
void InvokeForPair(Visitor &visitor, std::index_sequence<I3...>) {
  (InvokeOne<Visitor, Variant1, Variant2, Variant3, I1, I2, I3>(visitor), ...);
}

template <typename Visitor, typename Variant1, typename Variant2, typename Variant3, std::size_t I1, std::size_t I2,
          std::size_t I3>
void InvokeOne(Visitor &visitor) {
  using Value1 = std::variant_alternative_t<I1, Variant1>;
  using Value2 = std::variant_alternative_t<I2, Variant2>;
  using Value3 = std::variant_alternative_t<I3, Variant3>;
  const Variant1 value1{Value1{}};
  const Variant2 value2{Value2{}};
  const Variant3 value3{Value3{}};
  (void)std::visit(visitor, value1, value2, value3);
}

template <typename Visitor, typename Variant1, typename Variant2, typename Variant3, std::size_t I1, std::size_t... I2>
void InvokeForFirst(Visitor &visitor, std::index_sequence<I2...>) {
  (InvokeForPair<Visitor, Variant1, Variant2, Variant3, I1, I2>(
       visitor, std::make_index_sequence<std::variant_size_v<Variant3>>{}),
   ...);
}

template <typename Visitor, typename Variant1, typename Variant2, typename Variant3, std::size_t... I1>
void InvokeAllCombinations(Visitor &visitor, std::index_sequence<I1...>) {
  (InvokeForFirst<Visitor, Variant1, Variant2, Variant3, I1>(visitor,
                                                             std::make_index_sequence<std::variant_size_v<Variant2>>{}),
   ...);
}

template <typename Visitor, typename Variant1, typename Variant2, typename Variant3>
decltype(auto) VisitThree(Visitor &&visitor, Variant1 &&value1, Variant2 &&value2, Variant3 &&value3) {
  using V1 = RemoveCvRef<Variant1>;
  using V2 = RemoveCvRef<Variant2>;
  using V3 = RemoveCvRef<Variant3>;
  auto &visitor_ref = visitor;
  InvokeAllCombinations<Visitor, V1, V2, V3>(visitor_ref, std::make_index_sequence<std::variant_size_v<V1>>{});
  return std::visit(std::forward<Visitor>(visitor), std::forward<Variant1>(value1), std::forward<Variant2>(value2),
                    std::forward<Variant3>(value3));
}
}  // namespace autofuse_variant_coverage_detail

namespace std {
template <typename Visitor, typename Variant>
decltype(auto) autofuse_test_visit(Visitor &&visitor, Variant &&value) {
  return std::visit(std::forward<Visitor>(visitor), std::forward<Variant>(value));
}

template <typename Visitor, typename Variant1, typename Variant2>
decltype(auto) autofuse_test_visit(Visitor &&visitor, Variant1 &&value1, Variant2 &&value2) {
  return std::visit(std::forward<Visitor>(visitor), std::forward<Variant1>(value1), std::forward<Variant2>(value2));
}

template <typename Visitor, typename Variant1, typename Variant2, typename Variant3>
decltype(auto) autofuse_test_visit(Visitor &&visitor, Variant1 &&value1, Variant2 &&value2, Variant3 &&value3) {
  return ::autofuse_variant_coverage_detail::VisitThree(std::forward<Visitor>(visitor), std::forward<Variant1>(value1),
                                                        std::forward<Variant2>(value2), std::forward<Variant3>(value3));
}
}  // namespace std

// The production source uses std::visit. The macro is enabled only for the
// force-included test translation unit and leaves all other targets unchanged.
#define visit autofuse_test_visit

#endif  // AUTOFUSE_APPLY_ADAMD_VARIANT_COVERAGE_HOOK_H_
