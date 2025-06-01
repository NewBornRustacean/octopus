use std::marker::PhantomData;

pub trait ArmLogic<Context, Output>: Send + Sync {
    fn execute(&self, context: &Context) -> Output;
}

pub struct FnArmLogic<F, Context, Output>
where
    F: Fn(&Context) -> Output + Send + Sync,
{
    func: F,
    _phantom: PhantomData<(Context, Output)>,
}

impl<F, Context, Output: Clone + Send + Sync> FnArmLogic<F, Context, Output>
where
    F: Fn(&Context) -> Output + Send + Sync,
{
    pub fn new(func: F) -> Self {
        Self {
            func,
            _phantom: PhantomData,
        }
    }
}

impl<F, Context: Send + Sync, Output: Clone + Send + Sync> ArmLogic<Context, Output>
    for FnArmLogic<F, Context, Output>
where
    F: Fn(&Context) -> Output + Send + Sync,
{
    fn execute(&self, context: &Context) -> Output {
        (self.func)(context)
    }
}

pub struct ConstantLogic<Context, Output: Clone + Send + Sync> {
    value: Output,
    _phantom: PhantomData<Context>,
}

impl<Context: Send + Sync, Output: Clone + Send + Sync> ConstantLogic<Context, Output> {
    pub fn new(value: Output) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }
}

impl<Context: Send + Sync, Output: Clone + Send + Sync> ArmLogic<Context, Output>
    for ConstantLogic<Context, Output>
{
    fn execute(&self, _context: &Context) -> Output {
        self.value.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // Test context types
    #[derive(Debug, Clone, PartialEq)]
    struct TestContext {
        value: i32,
        name: String,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct ComplexContext {
        numbers: Vec<i32>,
        flag: bool,
    }

    mod fn_arm_logic_tests {
        use super::*;

        #[test]
        fn test_simple_function() {
            let logic = FnArmLogic::new(|ctx: &TestContext| ctx.value * 2);
            let context = TestContext {
                value: 42,
                name: "test".to_string(),
            };
            assert_eq!(logic.execute(&context), 84);
        }

        #[test]
        fn test_string_manipulation() {
            let logic = FnArmLogic::new(|ctx: &TestContext| format!("{}: {}", ctx.name, ctx.value));
            let context = TestContext {
                value: 42,
                name: "test".to_string(),
            };
            assert_eq!(logic.execute(&context), "test: 42");
        }

        #[test]
        fn test_complex_context() {
            let logic = FnArmLogic::new(|ctx: &ComplexContext| {
                if ctx.flag {
                    ctx.numbers.iter().sum::<i32>()
                } else {
                    0
                }
            });

            let context = ComplexContext {
                numbers: vec![1, 2, 3, 4],
                flag: true,
            };
            assert_eq!(logic.execute(&context), 10);

            let context = ComplexContext {
                numbers: vec![1, 2, 3, 4],
                flag: false,
            };
            assert_eq!(logic.execute(&context), 0);
        }

        #[test]
        fn test_closure_capture() {
            let multiplier = 2;
            let logic = FnArmLogic::new(move |ctx: &TestContext| ctx.value * multiplier);
            let context = TestContext {
                value: 21,
                name: "test".to_string(),
            };
            assert_eq!(logic.execute(&context), 42);
        }

        #[test]
        fn test_thread_safety() {
            let logic = Arc::new(FnArmLogic::new(|ctx: &TestContext| ctx.value + 1));
            let context = Arc::new(TestContext {
                value: 41,
                name: "test".to_string(),
            });

            let logic_clone = Arc::clone(&logic);
            let context_clone = Arc::clone(&context);

            let handle = std::thread::spawn(move || logic_clone.execute(&context_clone));

            assert_eq!(handle.join().unwrap(), 42);
        }
    }

    mod constant_logic_tests {
        use super::*;

        #[test]
        fn test_numeric_constant() {
            let logic = ConstantLogic::new(42);
            let context = TestContext {
                value: 100,
                name: "test".to_string(),
            };
            assert_eq!(logic.execute(&context), 42);
        }

        #[test]
        fn test_string_constant() {
            let logic = ConstantLogic::new("constant".to_string());
            let context = TestContext {
                value: 42,
                name: "test".to_string(),
            };
            assert_eq!(logic.execute(&context), "constant".to_string());
        }

        #[test]
        fn test_complex_type_constant() {
            let logic = ConstantLogic::new(ComplexContext {
                numbers: vec![1, 2, 3],
                flag: true,
            });
            let context = TestContext {
                value: 42,
                name: "test".to_string(),
            };
            let result = logic.execute(&context);
            assert_eq!(result.numbers, vec![1, 2, 3]);
            assert!(result.flag);
        }

        #[test]
        fn test_thread_safety() {
            let logic = Arc::new(ConstantLogic::new(42));
            let context = Arc::new(TestContext {
                value: 100,
                name: "test".to_string(),
            });

            let logic_clone = Arc::clone(&logic);
            let context_clone = Arc::clone(&context);

            let handle = std::thread::spawn(move || logic_clone.execute(&context_clone));

            assert_eq!(handle.join().unwrap(), 42);
        }

        #[test]
        fn test_multiple_executions() {
            let logic = ConstantLogic::new(42);
            let context1 = TestContext {
                value: 100,
                name: "test1".to_string(),
            };
            let context2 = TestContext {
                value: 200,
                name: "test2".to_string(),
            };

            // Same constant value regardless of context
            assert_eq!(logic.execute(&context1), 42);
            assert_eq!(logic.execute(&context2), 42);
        }
    }

    mod trait_object_tests {
        use super::*;

        #[test]
        fn test_trait_object_dispatch() {
            // Create different logic implementations
            let fn_logic: Box<dyn ArmLogic<TestContext, i32>> =
                Box::new(FnArmLogic::new(|ctx: &TestContext| ctx.value * 2));
            let const_logic: Box<dyn ArmLogic<TestContext, i32>> = Box::new(ConstantLogic::new(42));

            let context = TestContext {
                value: 21,
                name: "test".to_string(),
            };

            // Test function-based logic
            assert_eq!(fn_logic.execute(&context), 42);

            // Test constant logic
            assert_eq!(const_logic.execute(&context), 42);
        }

        #[test]
        fn test_mixed_logic_types() {
            let logics: Vec<Box<dyn ArmLogic<TestContext, String>>> = vec![
                Box::new(FnArmLogic::new(|ctx: &TestContext| {
                    format!("fn: {}", ctx.value)
                })),
                Box::new(ConstantLogic::new("constant".to_string())),
            ];

            let context = TestContext {
                value: 42,
                name: "test".to_string(),
            };

            assert_eq!(logics[0].execute(&context), "fn: 42");
            assert_eq!(logics[1].execute(&context), "constant");
        }
    }
}
