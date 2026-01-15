from verifiers import SingleGradientVerifier, ExpectedGradientVerifier


def main():
    print("=== STARTING GRADIENT VERIFICATION SUITE ===\n")

    # 1. Verify Single Function f(t)
    v_f = SingleGradientVerifier()
    v_f.run_stress_test()
    v_f.report()

    # 2. Verify Expected Function g(t)
    v_g = ExpectedGradientVerifier()
    v_g.run_stress_test()
    v_g.report()


if __name__ == "__main__":
    main()
