parallelExecution in Global := false

lazy val Tape = project.dependsOn(logs, ProjectRef(file("RAII.scala"), "asynchronous"))

lazy val tapefactories =
  project.dependsOn(Tape, ProjectRef(file("RAII.scala"), "asynchronous"), Caller)

lazy val Caller = project

includeFilter in unmanagedSources := (includeFilter in unmanagedSources).value && new SimpleFileFilter(_.isFile)

lazy val `differentiable-Float` = project.dependsOn(`differentiable-Any`, tapefactories, math)

lazy val `differentiable-INDArray` = project.dependsOn(`differentiable-Double`)

val FloatRegex = """(?i:float)""".r

lazy val `differentiable-Double` = project
  .dependsOn(`differentiable-Any`, tapefactories, math)
  .settings(sourceGenerators in Compile += Def.task {
    for {
      floatFile <- (unmanagedSources in Compile in `differentiable-Float`).value
      floatDirectory <- (unmanagedSourceDirectories in Compile in `differentiable-Float`).value
      relativeFile <- floatFile.relativeTo(floatDirectory)
    } yield {
      val floatSource = IO.read(floatFile, scala.io.Codec.UTF8.charSet)

      val doubleSource = FloatRegex.replaceAllIn(floatSource, { m =>
        m.matched match {
          case "Float" => "Double"
          case "float" => "double"
        }
      })

      val outputFile = (sourceManaged in Compile).value / relativeFile.getPath
      IO.write(outputFile, doubleSource, scala.io.Codec.UTF8.charSet)
      outputFile
    }
  }.taskValue)

lazy val Lift = project.dependsOn(Tape, ProjectRef(file("RAII.scala"), "asynchronous"))

lazy val math = project.dependsOn(Lift)

lazy val `differentiable-Any` = project.dependsOn(Tape, ProjectRef(file("RAII.scala"), "asynchronous"))

lazy val logs = project.dependsOn(Caller)

lazy val `differentiable` = project.dependsOn(`differentiable-Float`, `differentiable-INDArray`)

publishArtifact := false

lazy val unidoc = project
  .enablePlugins(StandaloneUnidoc, TravisUnidocTitle)
  .settings(
    UnidocKeys.unidocProjectFilter in ScalaUnidoc in UnidocKeys.unidoc := {
      import Ordering.Implicits._
      if (VersionNumber(scalaVersion.value).numbers >= Seq(2, 12)) {
        inAggregates(LocalRootProject) -- inProjects(`differentiable-INDArray`)
      } else {
        inAggregates(LocalRootProject)
      }
    },
    addCompilerPlugin("org.spire-math" %% "kind-projector" % "0.9.3"),
    addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full),
    scalacOptions += "-Xexperimental"
  )

organization in ThisBuild := "com.thoughtworks.deeplearning"

crossScalaVersions := Seq("2.11.11", "2.12.2")

fork in ThisBuild in Test := true
