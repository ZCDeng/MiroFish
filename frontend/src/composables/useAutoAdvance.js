import { ref, watch, onUnmounted } from 'vue'

/**
 * 自动推进步骤的 composable
 * 当 conditionFn 返回 true 且 autoAdvance=true 时，倒计时后自动调用 action
 *
 * @param {Function} conditionFn  - 返回 boolean 的函数，满足时触发倒计时
 * @param {Function} action       - 自动触发的回调函数
 * @param {Object}   options
 * @param {Ref}      options.enabled     - 是否启用（对应 autoAdvance prop）
 * @param {number}   options.delay       - 倒计时秒数，默认 3
 * @param {Array}    options.watchSources - 需要监听的响应式数据源
 */
export function useAutoAdvance(conditionFn, action, { enabled, delay = 3, watchSources = [] } = {}) {
  const countdown = ref(0)
  const active = ref(false)
  let timer = null

  const cancel = () => {
    if (timer) {
      clearInterval(timer)
      timer = null
    }
    countdown.value = 0
    active.value = false
  }

  const start = () => {
    if (active.value) return
    active.value = true
    countdown.value = delay
    timer = setInterval(() => {
      countdown.value--
      if (countdown.value <= 0) {
        cancel()
        action()
      }
    }, 1000)
  }

  watch(
    watchSources,
    () => {
      if (enabled?.value && conditionFn() && !active.value) {
        start()
      }
    },
    { immediate: true }
  )

  onUnmounted(cancel)

  return { countdown, active, cancel }
}
