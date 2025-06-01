use octopus::{
    algorithm::{EpsilonGreedy, error::BanditError},
    common::{
        arm::NumericArm,
        error::StateError,
        reward::{MeanAggregator, NumericReward},
        state::StateStore,
    },
};
use std::sync::Arc;
use std::thread;

#[test]
fn test_basic_usage() {
    // Create state store and algorithm
    let state: StateStore<NumericArm, MeanAggregator> = StateStore::new();
    let bandit = EpsilonGreedy::new(0.1).unwrap();

    // Add arms with different true means
    let arm1 = NumericArm::new("arm1".to_string()); // mean = 0.5
    let arm2 = NumericArm::new("arm2".to_string()); // mean = 0.7
    let arm3 = NumericArm::new("arm3".to_string()); // mean = 0.3

    state.add_arm(arm1.clone(), MeanAggregator::new()).unwrap();
    state.add_arm(arm2.clone(), MeanAggregator::new()).unwrap();
    state.add_arm(arm3.clone(), MeanAggregator::new()).unwrap();

    println!("\nInitial state:");
    state.print_state();

    // Simulate 1000 pulls
    let n_pulls = 1000;
    let mut arm1_pulls = 0;
    let mut arm2_pulls = 0;
    let mut arm3_pulls = 0;

    for i in 0..n_pulls {
        let selected_arm = bandit.select_arm(&state).unwrap();

        // Generate reward based on true mean + some noise
        let reward = match selected_arm.id {
            id if id == arm1.id => NumericReward::new(0.5 + rand::random::<f64>() * 0.1).unwrap(),
            id if id == arm2.id => NumericReward::new(0.7 + rand::random::<f64>() * 0.1).unwrap(),
            id if id == arm3.id => NumericReward::new(0.3 + rand::random::<f64>() * 0.1).unwrap(),
            _ => panic!("Unknown arm selected"),
        };

        // Update state with reward
        state.update(selected_arm.clone(), reward).unwrap();

        // Count pulls
        match selected_arm.id {
            id if id == arm1.id => arm1_pulls += 1,
            id if id == arm2.id => arm2_pulls += 1,
            id if id == arm3.id => arm3_pulls += 1,
            _ => panic!("Unknown arm selected"),
        };

        // Print state every 200 pulls
        if (i + 1) % 200 == 0 {
            println!("\nState after {} pulls:", i + 1);
            state.print_state();
        }
    }

    println!("\nFinal state:");
    state.print_state();

    // Verify that arm2 (highest mean) was pulled most often
    assert!(arm2_pulls > arm1_pulls);
    assert!(arm2_pulls > arm3_pulls);

    // Verify that exploration still happened (no arm was completely ignored)
    assert!(arm1_pulls > 0);
    assert!(arm2_pulls > 0);
    assert!(arm3_pulls > 0);

    // Verify final estimates are close to true means
    let arm1_estimate = state.estimate(arm1).unwrap();
    let arm2_estimate = state.estimate(arm2).unwrap();
    let arm3_estimate = state.estimate(arm3).unwrap();

    assert!((arm1_estimate - 0.5).abs() < 0.1);
    assert!((arm2_estimate - 0.7).abs() < 0.1);
    assert!((arm3_estimate - 0.3).abs() < 0.1);
}

#[test]
fn test_concurrent_access() {
    let state: Arc<StateStore<NumericArm, MeanAggregator>> = Arc::new(StateStore::new());
    let bandit = Arc::new(EpsilonGreedy::new(0.1).unwrap());

    // Add arms
    let arm1 = NumericArm::new("arm1".to_string());
    let arm2 = NumericArm::new("arm2".to_string());
    let arm3 = NumericArm::new("arm3".to_string());

    state.add_arm(arm1.clone(), MeanAggregator::new()).unwrap();
    state.add_arm(arm2.clone(), MeanAggregator::new()).unwrap();
    state.add_arm(arm3.clone(), MeanAggregator::new()).unwrap();

    // Create multiple threads that pull arms concurrently
    let n_threads = 4;
    let pulls_per_thread = 250;
    let handles: Vec<_> = (0..n_threads)
        .map(|_| {
            let state = Arc::clone(&state);
            let bandit = Arc::clone(&bandit);
            thread::spawn(move || {
                for _ in 0..pulls_per_thread {
                    let selected_arm = bandit.select_arm(&state).unwrap();
                    let reward = NumericReward::new(rand::random::<f64>()).unwrap();
                    state.update(selected_arm, reward).unwrap();
                }
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify that all arms were pulled
    assert!(state.pulls(arm1).unwrap() > 0);
    assert!(state.pulls(arm2).unwrap() > 0);
    assert!(state.pulls(arm3).unwrap() > 0);

    // Verify total pulls
    assert_eq!(state.total_pulls(), n_threads * pulls_per_thread);
}

#[test]
fn test_exploration_exploitation_balance() {
    let state: StateStore<NumericArm, MeanAggregator> = StateStore::new();
    let epsilon = 0.2;
    let bandit = EpsilonGreedy::new(epsilon).unwrap();

    // Add arms with very different means
    let arm1 = NumericArm::new("arm1".to_string()); // mean = 0.1
    let arm2 = NumericArm::new("arm2".to_string()); // mean = 0.9

    state.add_arm(arm1.clone(), MeanAggregator::new()).unwrap();
    state.add_arm(arm2.clone(), MeanAggregator::new()).unwrap();

    println!("\nInitial state:");
    state.print_state();

    // Initial pulls to establish estimates
    for i in 0..10 {
        let selected_arm = bandit.select_arm(&state).unwrap();
        let reward = if selected_arm.id == arm1.id {
            NumericReward::new(0.1).unwrap()
        } else {
            NumericReward::new(0.9).unwrap()
        };
        state.update(selected_arm, reward).unwrap();

        if (i + 1) % 2 == 0 {
            println!("\nState after {} initial pulls:", i + 1);
            state.print_state();
        }
    }

    // Count selections over many pulls
    let n_pulls = 1000;
    let mut arm1_selections = 0;
    let mut arm2_selections = 0;

    for _ in 0..n_pulls {
        let selected_arm = bandit.select_arm(&state).unwrap();
        if selected_arm.id == arm1.id {
            arm1_selections += 1;
        } else {
            arm2_selections += 1;
        }
    }

    // Calculate exploration rate
    let exploration_rate = arm1_selections as f64 / n_pulls as f64;

    // Exploration rate should be roughly epsilon/2 (since we have 2 arms)
    // Allow for some variance due to randomness
    let expected_exploration = epsilon / 2.0;
    assert!((exploration_rate - expected_exploration).abs() < 0.1);

    // Best arm (arm2) should be selected more often
    assert!(arm2_selections > arm1_selections);

    println!("\nFinal state:");
    state.print_state();
}

#[test]
fn test_error_cases() {
    let state: StateStore<NumericArm, MeanAggregator> = StateStore::new();
    let bandit = EpsilonGreedy::new(0.1).unwrap();

    // Test with empty state
    assert!(matches!(
        bandit.select_arm(&state),
        Err(BanditError::StateError(StateError::NoArmsAvailable))
    ));

    // Test invalid epsilon values
    assert!(matches!(
        EpsilonGreedy::new(-0.1),
        Err(BanditError::InvalidEpsilon(-0.1))
    ));
    assert!(matches!(
        EpsilonGreedy::new(1.1),
        Err(BanditError::InvalidEpsilon(1.1))
    ));
}
